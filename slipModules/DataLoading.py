#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 14:11:52 2023

@author: fiftak
"""

import os
import numpy as np
from PIL import Image
from torch.utils import data
import torch

# from tqdm import tqdm

import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


## ------------------- label conversion tools ------------------ ##
def labels2cat(label_encoder, list):
    return label_encoder.transform(list)


def labels2onehot(OneHotEncoder, label_encoder, list):
    return OneHotEncoder.transform(
        label_encoder.transform(list).reshape(-1, 1)
    ).toarray()


def onehot2labels(label_encoder, y_onehot):
    return label_encoder.inverse_transform(np.where(y_onehot == 1)[1]).tolist()


def cat2labels(label_encoder, y_cat):
    return label_encoder.inverse_transform(y_cat).tolist()


## ---------------------- Dataloaders ---------------------- ##
# for 3DCNN
class Dataset_3DCNN(data.Dataset):
    "Characterizes a dataset for PyTorch"

    def __init__(
        self,
        data_path: str,
        folders: str,
        labels: list,
        frames: list[int],
        transform=None,
    ):
        "Initialization"
        self.data_path = data_path
        self.labels = labels
        self.folders = folders
        self.transform = transform
        self.frames = frames
        # self.folder_list = ['null']*len(self.folders)

    def __len__(self):
        "Denotes the total number of samples"
        return len(self.folders)

    # #IMAGE - ORIGINAL
    # def read_images(self, path, selected_folder):
    #     X = []
    #     # print('path:', path)
    #     # print('selected_folder:', selected_folder)
    #     folder_path = path+'/'+selected_folder
    #     for i in sorted(os.listdir(folder_path))[:len(self.frames)+1]:
    #         image = Image.open(os.path.join(folder_path, str(i) )).convert('L')

    #         if self.transform is not None:
    #             image = self.transform(image)
    #         X.append(image.squeeze_(0))

    #     X = torch.stack(X, dim=0)
    #     return X

    # IMAGE - DIFFERENCE
    def read_images(self, path: str, selected_folder: str):
        "Reads data images, transform and return in tensor form"
        X = []
        folder_path = path + selected_folder
        images = sorted(os.listdir(folder_path))
        image0 = self.transform(
            Image.open(os.path.join(folder_path, str(images[0]))).convert("L")
        )

        for i in images[: len(self.frames)]:  # from 1st to selected_frames
            image = Image.open(os.path.join(folder_path, str(i))).convert("L")

            if self.transform is not None:
                image = self.transform(image)
                image -= image0
            X.append(image.squeeze_(0))

        X = torch.stack(X, dim=0)
        return X

    def __getitem__(self, index):
        "Generates one sample of data"
        # Select sample
        folder = self.folders[index]
        # # Save sample for future reference
        # self.folder_list[index] = folder

        # Load data
        X = self.read_images(self.data_path, folder).unsqueeze_(
            0
        )  # (input) spatial images
        y = torch.LongTensor(
            [self.labels[index]]
        )  # (labels) LongTensor are for int64 instead of FloatTensor

        return X, y


## ---------------------- Create_loaders ---------------------- ##


def create_loaders(
    data_path: int,
    selected_frames: list[int],
    img_size: list[int],
    batch_size: int,
    random_state: int = None):
    # Get a list of all labels in the data directory
    labels = [name for name in os.listdir(data_path)
        if os.path.isdir(os.path.join(data_path, name))]

    img_x, img_y = img_size

    x_list = []
    y_list = []

    # Iterate over each label directory
    for label in labels:
        class_path = os.path.join(data_path, label)
        # List sequences(each containing N images) within the current label directory
        for x in os.listdir(class_path):
            if os.path.isdir(os.path.join(class_path, x)):
                x_list.append("/" + os.path.join(label, x)), y_list.append(label)
    # print("Sequences :", x_list)
    # print('labels :', y_list)

    # Encode the y_list
    label_encoder = LabelEncoder()
    label_encoder.fit(y_list)
    y_list = labels2cat(label_encoder, y_list)

    # fraction of training set to be used for validation
    validation_size = 0.3

    # train, test split
    if random_state == None:
        x_train, x_test, y_train, y_test = train_test_split(
            x_list, y_list, test_size=0.2
        )
    else:
        x_train, x_test, y_train, y_test = train_test_split(
            x_list, y_list, test_size=0.2, random_state=random_state
        )

    # image transformation
    transform = transforms.Compose(
        [
            transforms.Resize([img_x, img_y]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ]
    )

    train_set, test_set = Dataset_3DCNN(
        data_path, x_train, y_train, selected_frames, transform=transform
    ), Dataset_3DCNN(data_path, x_test, y_test, selected_frames, transform=transform)

    # split into training and validation batches
    num_train = len(train_set)
    indices = list(range(num_train))
    np.random.shuffle(indices)
    split = int(np.floor(validation_size * num_train))
    train_idx, valid_idx = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    # loading train, validation and test data

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, sampler=train_sampler, num_workers=0
    )
    valid_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, sampler=valid_sampler, num_workers=0
    )
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size, num_workers=0
    )

    return train_loader, valid_loader, test_loader


# ## -------------------- (reload) model prediction ---------------------- ##
# def Conv3d_final_prediction(model, device, loader):
#     model.eval()

#     all_y_pred = []
#     with torch.no_grad():
#         for batch_idx, (X, y) in enumerate(tqdm(loader)):
#             # distribute data to device
#             X = X.to(device)
#             output = model(X)
#             y_pred = output.max(1, keepdim=True)[1]  # location of max log-probability as prediction
#             all_y_pred.extend(y_pred.cpu().data.squeeze().numpy().tolist())

#     return all_y_pred


# ## -------------------- end of model prediction ---------------------- ##
