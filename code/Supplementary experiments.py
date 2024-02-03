import torch
import argparse
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from our_attention import CustomAttention
from MyModel import resnet18, cls_head, LSTMModel, FeatureReduction
from data_process import CustomDataset, load_dataset, load_csv_dataset
from sklearn.metrics import confusion_matrix

parser = argparse.ArgumentParser(description="Dual flow structure baseline")
parser.add_argument("--lr_backbone", type=float, default=1e-5, help="Learning rate for the optimizer of backbone")
parser.add_argument("--lr_cls_base", type=float, default=1e-5,
                    help="Learning rate for the optimizer of source feature decouple")
parser.add_argument("--lr_lstm", type=float, default=1e-5, help="Learning rate for the optimizer of lstm model")
parser.add_argument("--lr_attention", type=float, default=1e-5,
                    help="Learning rate for the optimizer of classification head")

parser.add_argument("--alpha", type=float, default=0.01, help="Weight for cls_f1: rgb")
parser.add_argument("--beta", type=float, default=0.005, help="Weight for cls_f2: skletopn")
parser.add_argument("--gamma", type=float, default=1e-04, help="Weight for Attention loss")
parser.add_argument("--flag", type=bool, default=False, help="Whether to enable similarity selection")
parser.add_argument("--rank", type=int, default=1, help="important feature index")

args = parser.parse_args()

print(args)

X_rgb_train, X_rgb_val, y_rgb_train, y_rgb_val = load_dataset("/root/autodl-fs/tsp_code/rgb5")

X_skl_train, X_skl_val, y_skl_train, y_skl_val = load_dataset("/root/autodl-fs/tsp_code/ske5")

X_motion_train, X_motion_val, y_motion_train, y_motion_val = load_csv_dataset(
    "/root/autodl-fs/tsp_code/motion_data.csv")

train_data = CustomDataset(X_rgb_train, y_rgb_train, X_skl_train, y_skl_train, X_motion_train, y_motion_train)

val_data = CustomDataset(X_rgb_val, y_rgb_val, X_skl_val, y_skl_val, X_motion_val, y_motion_val)

train_loader = DataLoader(train_data, batch_size=16, shuffle=True, drop_last=True)
val_loader = DataLoader(val_data, batch_size=16, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

feature_dim = 512
num_heads = 8

resnet_18 = resnet18()
feature_reduction = FeatureReduction(feature_dim * 2, feature_dim)
lstm = LSTMModel(input_size=4, hidden_size=256, output_size=512)
cls_base = cls_head(num_class=4)  # Directly used as a classifier on features
attention = CustomAttention(feature_dim, num_heads)
# cls_attention = cls_head() # Classifier used on features after Attention

optimizer_backbone = torch.optim.Adam(resnet_18.parameters(), lr=args.lr_backbone)
optimizer_cls_base = torch.optim.Adam(cls_base.parameters(), lr=args.lr_cls_base)
optimizer_attention = torch.optim.Adam(attention.parameters(), lr=args.lr_attention)
optimizer_lstm = torch.optim.Adam(lstm.parameters(), lr=args.lr_lstm)
optimizer_reduction = torch.optim.Adam(feature_reduction.parameters(), lr=args.lr_lstm)


# optimizer_cls_attention = torch.optim.Adam(cls_attention.parameters(), lr=args.lr_cls_attention)

def count_parameters(models):
    return sum(p.numel() for model in models for p in model.parameters() if p.requires_grad)


def count_module_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)


def train(model, cls_base, attention, lstm, feature_reduction, dataloader, optimizer_backbone, optimizer_cls_base,
          optimizer_attention, device, optimizer_lstm, optimizer_reduction):
    model.train().to(device)
    lstm.train().to(device)
    cls_base.train().to(device)
    attention.train().to(device)
    feature_reduction.train().to(device)
    # cls_attention.train()

    f3_labels_all = []
    f3_preds_all = []

    total_loss = 0
    correct_base_f1 = 0
    correct_base_f2 = 0
    correct_base_f3 = 0
    correct_attention = 0
    rgb_l, skl_l = 0, 0

    for data in dataloader:
        rgb_imgs, rgb_labels, skl_imgs, skl_labels, motion_feature, motion_label = data
        rgb_imgs, rgb_labels, skl_imgs, skl_labels, motion_feature, motion_label = rgb_imgs.to(device), rgb_labels.to(
            device), skl_imgs.to(
            device), skl_labels.to(device), motion_feature.to(device), motion_label.to(device)

        rgb_l += rgb_imgs.shape[0]
        skl_l += skl_imgs.shape[0]

        optimizer_backbone.zero_grad()
        optimizer_cls_base.zero_grad()
        optimizer_attention.zero_grad()
        optimizer_lstm.zero_grad()
        optimizer_reduction.zero_grad()

        f3_labels_all.extend(motion_label.cpu().numpy())

        # Get the features from the backbone model
        f1 = model(rgb_imgs)
        f2 = model(skl_imgs)
        f3 = lstm(motion_feature)

        # f2.shape [16, 512]
        # f3.shape [16, 512]

        feature_concatenate = torch.cat([f2, f3], dim=1)

        feature_concatenate = feature_reduction(feature_concatenate)

        # Classifier on original features
        logits_base_f1 = cls_base(f1)
        logits_base_f2 = cls_base(f2)
        logits_base_f3 = cls_base(f3)

        loss_base_f1 = F.cross_entropy(logits_base_f1, rgb_labels)
        loss_base_f2 = F.cross_entropy(logits_base_f2, skl_labels)
        loss_base_f3 = F.cross_entropy(logits_base_f3, motion_label)

        # Apply the attention mechanism

        if args.flag:
            norm_f1 = F.normalize(f1, p=2, dim=1)
            norm_feature_concatenate = F.normalize(feature_concatenate, p=2, dim=1)
            similarity_matrix = torch.mm(norm_f1, norm_feature_concatenate.t())
            mean_similarities = torch.mean(similarity_matrix, dim=1)
            most_similar_index = torch.argmax(mean_similarities)
            top_similarities, top_indices = torch.topk(mean_similarities, k=args.rank)
            selected_f1 = f1[most_similar_index]
            f1 = selected_f1.unsqueeze(0).expand_as(f1)

        # if args.flag:
        #     norm_f1 = F.normalize(f1, p=2, dim=1)
        #     norm_feature_concatenate = F.normalize(feature_concatenate, p=2, dim=1)
        #     similarity_matrix = torch.mm(norm_f1, norm_feature_concatenate.t())
        #     mean_similarities = torch.mean(similarity_matrix, dim=1)
        #     most_similar_index = torch.argmax(mean_similarities)
        #     top_similarities, top_indices = torch.topk(mean_similarities, k=args.rank)
        #     selected_con = feature_concatenate[most_similar_index]
        #     feature_concatenate = selected_con.unsqueeze(0).expand_as(feature_concatenate)

        fused_features = attention(f1, feature_concatenate)

        # shortcut
        fused_features += f3

        # Classifier on features after Attention
        logits_attention = cls_base(fused_features)
        loss_attention = F.cross_entropy(logits_attention, motion_label)

        # Combine the losses
        loss = loss_base_f1 * args.alpha + loss_base_f2 * args.beta + loss_attention
        total_loss += loss.item()

        # Backpropagation
        loss.backward()
        optimizer_backbone.step()
        optimizer_cls_base.step()
        optimizer_reduction.step()
        optimizer_attention.step()
        optimizer_lstm.step()

        # Calculate accuracy
        _, preds_base_f1 = torch.max(logits_base_f1, 1)
        _, preds_base_f2 = torch.max(logits_base_f2, 1)
        _, preds_base_f3 = torch.max(logits_base_f3, 1)

        f3_preds_all.extend(preds_base_f3.cpu().numpy())

        _, preds_attention = torch.max(logits_attention, 1)

        correct_base_f1 += (preds_base_f1 == rgb_labels).sum().item()
        correct_base_f2 += (preds_base_f2 == skl_labels).sum().item()
        correct_base_f3 += (preds_base_f3 == motion_label).sum().item()

        correct_attention += (preds_attention == motion_label).sum().item()

    avg_loss = total_loss / len(dataloader)
    accuracy_base_f1 = correct_base_f1 / rgb_l
    accuracy_base_f2 = correct_base_f2 / skl_l

    accuracy_base_f3 = correct_base_f3 / skl_l

    accuracy_attention = correct_attention / (rgb_l)

    conf_matrix_f3 = confusion_matrix(f3_labels_all, f3_preds_all)
    class_accuracy_f3 = conf_matrix_f3.diagonal() / conf_matrix_f3.sum(axis=1)

    return avg_loss, accuracy_base_f1, accuracy_base_f2, class_accuracy_f3, accuracy_attention


def validate(model, cls_base, attention, dataloader, device):
    model.eval()
    cls_base.eval()
    attention.eval()
    lstm.eval()
    feature_reduction.eval()

    f3_labels_all = []
    f3_preds_all = []

    # cls_attention.eval()
    model.to(device)
    cls_base.to(device)
    attention.to(device)
    total_loss = 0
    correct_base_f1 = 0
    correct_base_f2 = 0
    correct_base_f3 = 0
    correct_attention = 0
    rgb_l, skl_l = 0, 0

    with torch.no_grad():
        for data in dataloader:
            rgb_imgs, rgb_labels, skl_imgs, skl_labels, motion_feature, motion_label = data
            rgb_imgs, rgb_labels, skl_imgs, skl_labels, motion_feature, motion_label = rgb_imgs.to(
                device), rgb_labels.to(device), skl_imgs.to(
                device), skl_labels.to(device), motion_feature.to(device), motion_label.to(device)

            rgb_l += rgb_imgs.shape[0]
            skl_l += skl_imgs.shape[0]

            # Get the features from the backbone model
            f1 = model(rgb_imgs)
            f2 = model(skl_imgs)
            f3 = lstm(motion_feature)

            feature_concatenate = torch.cat([f2, f3], dim=1)
            feature_concatenate = feature_reduction(feature_concatenate)

            # Classifier on original features

            logits_base_f1 = cls_base(f1)
            logits_base_f2 = cls_base(f2)
            logits_base_f3 = cls_base(f3)

            loss_base_f1 = F.cross_entropy(logits_base_f1, rgb_labels)
            loss_base_f2 = F.cross_entropy(logits_base_f2, skl_labels)
            loss_base_f3 = F.cross_entropy(logits_base_f3, motion_label)

            # Apply the attention mechanism
            # Apply the attention mechanism
            fused_features = attention(f1, feature_concatenate)

            # Classifier on features after Attention
            logits_attention = cls_base(fused_features)
            loss_attention = F.cross_entropy(logits_attention, motion_label)

            # Combine the losses
            loss = loss_base_f1 * args.alpha + loss_base_f2 * args.beta + loss_attention * args.gamma + loss_base_f3
            total_loss += loss.item()

            # Calculate accuracy
            _, preds_base_f1 = torch.max(logits_base_f1, 1)
            _, preds_base_f2 = torch.max(logits_base_f2, 1)
            f3_labels_all.extend(motion_label.cpu().numpy())

            _, preds_base_f3 = torch.max(logits_base_f3, 1)
            f3_preds_all.extend(preds_base_f3.cpu().numpy())

            _, preds_attention = torch.max(logits_attention, 1)

            correct_base_f1 += (preds_base_f1 == rgb_labels).sum().item()
            correct_base_f2 += (preds_base_f2 == skl_labels).sum().item()
            correct_base_f3 += (preds_base_f3 == motion_label).sum().item()

            correct_attention += (preds_attention == motion_label).sum().item()

    avg_loss = total_loss / len(dataloader)
    accuracy_base_f1 = correct_base_f1 / rgb_l
    accuracy_base_f2 = correct_base_f2 / skl_l
    accuracy_base_f3 = correct_base_f3 / skl_l
    accuracy_attention = correct_attention / (rgb_l)
    conf_matrix_f3 = confusion_matrix(f3_labels_all, f3_preds_all)
    class_accuracy_f3 = conf_matrix_f3.diagonal() / conf_matrix_f3.sum(axis=1)

    return avg_loss, accuracy_base_f1, accuracy_base_f2, class_accuracy_f3, accuracy_attention


# Training loop
num_epochs = 1
set_seed(42)

for epoch in range(num_epochs):
    train_loss, train_acc_base_f1, train_acc_base_f2, train_acc_base_f3, train_acc_attention = train(resnet_18,
                                                                                                     cls_base,
                                                                                                     attention, lstm,
                                                                                                     feature_reduction,
                                                                                                     train_loader,
                                                                                                     optimizer_backbone,
                                                                                                     optimizer_cls_base,
                                                                                                     optimizer_attention,
                                                                                                     device,
                                                                                                     optimizer_lstm,
                                                                                                     optimizer_reduction)
    val_loss, val_acc_base_f1, val_acc_base_f2, val_acc_base_f3, val_acc_attention = validate(resnet_18, cls_base,
                                                                                              attention, val_loader,
                                                                                              device)

    train_acc_base_f3_str = ', '.join([f'{acc * 100:.2f}%' for acc in train_acc_base_f3])
    val_acc_base_f3_str = ', '.join([f'{acc * 100:.2f}%' for acc in val_acc_base_f3])

    print(f'Epoch {epoch + 1}/{num_epochs}:')
    print(
        f'Train Loss: {train_loss:.4f}, Train Acc Base1: {train_acc_base_f1 * 100:.2f}%, Train Acc Base2: {train_acc_base_f2 * 100:.2f}%, Train Acc Attention: {train_acc_attention * 100:.2f}%')
    print(
        f'Val Loss: {val_loss:.4f}, Val Acc Base1: {val_acc_base_f1 * 100:.2f}%,  Val Acc Base2: {val_acc_base_f2 * 100:.2f}%,   Val Acc Attention: {val_acc_attention * 100:.2f}%')
    print(f"train f3: {train_acc_base_f3_str}")
    print(f"val f3: {val_acc_base_f3_str}")

# Models to count parameters
models = [resnet_18, lstm, cls_base, attention, feature_reduction]

total_params = count_parameters(models)
print(f"Total trainable parameters: {total_params}")
params_resnet_18 = count_module_parameters(resnet_18)
params_lstm = count_module_parameters(lstm)
params_cls_base = count_module_parameters(cls_base)
params_attention = count_module_parameters(attention)
params_feature_reduction = count_module_parameters(feature_reduction)

print(f"ResNet18 parameters: {params_resnet_18}")
print(f"LSTM parameters: {params_lstm}")
print(f"Classifier Head parameters: {params_cls_base}")
print(f"Attention Mechanism parameters: {params_attention}")
print(f"Feature Reduction parameters: {params_feature_reduction}")
