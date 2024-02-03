import torch
import random
import argparse
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from attention import MultiHeadSelfAttention
from MyModel import resnet18, cls_head
from data_process import BaseDataset, load_dataset

parser = argparse.ArgumentParser(description="Dual flow structure baseline")
parser.add_argument("--lr_backbone", type=float, default=1e-5, help="Learning rate for the optimizer of backbone")
parser.add_argument("--lr_cls_base", type=float, default=1e-5,
                    help="Learning rate for the optimizer of source feature decouple")
args = parser.parse_args()

X_rgb_train, X_rgb_val, y_rgb_train, y_rgb_val = load_dataset("./skeleton5")

train_data = BaseDataset(X_rgb_train, y_rgb_train)

val_data = BaseDataset(X_rgb_val, y_rgb_val)

train_loader = DataLoader(train_data, batch_size=2, shuffle=True, drop_last=True)
val_loader = DataLoader(val_data, batch_size=2, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

feature_dim = 512
num_heads = 8

resnet_18 = resnet18()
cls_base = cls_head()  # Directly used as a classifier on features

optimizer_backbone = torch.optim.Adam(resnet_18.parameters(), lr=args.lr_backbone)
optimizer_cls_base = torch.optim.Adam(cls_base.parameters(), lr=args.lr_cls_base)


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)


def train(model, cls_base, dataloader, optimizer_backbone, optimizer_cls_base, device):
    model.train().to(device)
    cls_base.train().to(device)

    total_loss = 0
    correct_base = 0

    for data in dataloader:
        rgb_imgs, rgb_labels = data

        rgb_imgs, rgb_labels = rgb_imgs.to(device), rgb_labels.to(device)

        optimizer_backbone.zero_grad()
        optimizer_cls_base.zero_grad()

        f = model(rgb_imgs)

        logits_base = cls_base(f)

        loss_base = F.cross_entropy(logits_base, rgb_labels)

        # Combine the losses
        loss = loss_base
        total_loss += loss.item()

        # Backpropagation
        loss.backward()
        optimizer_backbone.step()
        optimizer_cls_base.step()

        # Calculate accuracy
        _, preds_base = torch.max(logits_base, 1)
        correct_base += (preds_base == rgb_labels).sum().item()

    avg_loss = total_loss / len(dataloader)
    accuracy_base = correct_base / len(dataloader.dataset)

    return avg_loss, accuracy_base


def validate(model, cls_base, dataloader, device):
    model.eval()
    cls_base.eval()

    total_loss = 0
    correct_base = 0

    model.to(device)
    cls_base.to(device)

    with torch.no_grad():
        for data in dataloader:
            rgb_imgs, rgb_labels = data
            rgb_imgs, rgb_labels = rgb_imgs.to(device), rgb_labels.to(device)

            # Get the features from the backbone model
            f = model(rgb_imgs)

            # Classifier on original features

            logits_base = cls_base(f)

            loss_base = F.cross_entropy(logits_base, rgb_labels)

            # Combine the losses
            loss = loss_base
            total_loss += loss.item()

            # Calculate accuracy
            _, preds_base = torch.max(logits_base, 1)

            correct_base += (preds_base == rgb_labels).sum().item()

    avg_loss = total_loss / len(dataloader)
    accuracy_base = correct_base / len(dataloader.dataset)

    return avg_loss, accuracy_base


# Training loop
num_epochs = 300
set_seed(42)
for epoch in range(num_epochs):
    train_loss, train_acc_base = train(resnet_18, cls_base, train_loader, optimizer_backbone, optimizer_cls_base,
                                       device)
    val_loss, val_acc_base = validate(resnet_18, cls_base, val_loader, device)

    print(f'Epoch {epoch + 1}/{num_epochs}:')
    print(f'Train Loss: {train_loss:.4f}, Train Acc Base1: {train_acc_base * 100:.2f}%')
    print(f'Val Loss: {val_loss:.4f}, Val Acc Base1: {val_acc_base * 100:.2f}')


