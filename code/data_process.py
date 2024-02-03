import os
import cv2
import numpy as np
import torch
import h5py
import pandas as pd
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.model_selection import train_test_split



def save_to_hdf5(images, labels, file_name):
    with h5py.File(file_name, 'w') as f:
        f.create_dataset("images", data=images)
        f.create_dataset("labels", data=labels)


def load_from_hdf5(file_name):
    with h5py.File(file_name, 'r') as f:
        images = f["images"][:]
        labels = f["labels"][:]
    return images, labels


def load_dataset(folder_path, save_hdf5=True, train_hdf5="train_data.hdf5", val_hdf5="val_data.hdf5", mode="dual"):
    # 如果mode 是"dual" 就是双流结构，如果是其他的 就是默认的基础结构
    if save_hdf5 and os.path.exists(train_hdf5) and os.path.exists(val_hdf5) and mode == "dual":
        X_train, y_train = load_from_hdf5(train_hdf5)
        X_val, y_val = load_from_hdf5(val_hdf5)
    else:
        classes = sorted(os.listdir(folder_path))

        class_dict = {}

        for i, c in enumerate(classes):
            class_dict[c] = i

        images = []
        labels = []
        t = set()
        for c in classes:
            class_folder = os.path.join(folder_path, c)

            if not os.path.isdir(class_folder):
                continue

            for file in sorted(os.listdir(class_folder)):
                if file.startswith('.'):
                    continue

                image_path = os.path.join(class_folder, file)
                each_item = []

                for each_imgs in sorted(os.listdir(image_path)):
                    imgs_root = os.path.join(image_path, each_imgs)

                    image = cv2.imread(imgs_root)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    image = cv2.resize(image, (512, 288))
                    image = np.transpose(image, (2, 0, 1))
                    #改变图片的张量，[width,height,channels]-----→[channels,height,width]
                    t.add(image.shape)
                    each_item.append(image)

                labels.append(class_dict[c])
                images.append(np.concatenate(each_item, axis=0))
        print(t)
        images = np.stack(images)
        labels = np.array(labels)

        X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=3 / 10, shuffle=True)

        if save_hdf5:
            save_to_hdf5(X_train, y_train, train_hdf5)
            save_to_hdf5(X_val, y_val, val_hdf5)
            print("完成数据的保存")

    return X_train, X_val, y_train, y_val


def load_csv_dataset(fileroot):
    df = pd.read_csv(fileroot)
    # 提取特征和标签
    features = df[['0', '1', '2', '3']]
    labels = df['lable']

    # 按照3:7的比例划分训练数据和验证数据
    features_train, features_valid, labels_train, labels_valid = train_test_split(features, labels, test_size=3/10,
                                                                                  shuffle=True)

    features_train = features_train.values
    features_valid = features_valid.values
    labels_train = labels_train.values
    labels_valid = labels_valid.values

    return features_train, features_valid, labels_train, labels_valid

class CustomDataset(Dataset):
    def __init__(self, rgb_images, rgb_labels, skl_images, skl_labels, motion_feature=None, motion_label=None):
        self.rgb_images = rgb_images

        self.rgb_labels = rgb_labels

        self.skl_images = skl_images

        self.skl_labels = skl_labels

        self.motion_feature = motion_feature

        self.motion_label = motion_label

    def __len__(self):
        return len(self.rgb_labels)

    def __getitem__(self, index):
        index_rgb = index % len(self.rgb_labels)

        index_skl = index % len(self.skl_labels)

        index_motion = index % len(self.motion_label)

        rgb_image = torch.from_numpy(self.rgb_images[index_rgb]).float()

        rgb_label = torch.tensor(self.rgb_labels[index_rgb], dtype=torch.long)

        skl_image = torch.from_numpy(self.skl_images[index_skl]).float()

        skl_label = torch.tensor(self.skl_labels[index_skl], dtype=torch.long)

        if self.motion_label is not None and self.motion_feature is not None:

            motion_feature = torch.from_numpy(self.motion_feature[index_motion]).float()

            motion_label = torch.tensor(self.motion_label[index_motion], dtype=torch.long)

            return rgb_image, rgb_label, skl_image, skl_label, motion_feature, motion_label

        return rgb_image, rgb_label, skl_image, skl_label


class BaseDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images

        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = torch.from_numpy(self.images[index]).float()

        label = torch.tensor(self.labels[index], dtype=torch.long)

        return image, label



class TripleDataset(Dataset):
    def __init__(self, rgb_images, rgb_labels, skl_images, skl_labels):
        self.rgb_images = rgb_images

        self.rgb_labels = rgb_labels

        self.skl_images = skl_images

        self.skl_labels = skl_labels

    def __len__(self):
        return len(self.rgb_labels) + len(self.skl_labels)

    def __getitem__(self, index):
        index_rgb = index % len(self.rgb_labels)

        index_skl = index % len(self.skl_labels)

        rgb_image = torch.from_numpy(self.rgb_images[index_rgb]).float()

        rgb_label = torch.tensor(self.rgb_labels[index_rgb], dtype=torch.long)

        skl_image = torch.from_numpy(self.skl_images[index_skl]).float()

        skl_label = torch.tensor(self.skl_labels[index_skl], dtype=torch.long)

        return rgb_image, rgb_label, skl_image, skl_label

