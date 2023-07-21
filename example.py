#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 10:28:14 2023

@author: fiftak
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import sklearn

# Add my modules' path and import
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir + "/slipModules")

from DataLoading import create_loaders
from Model import CNN3D
from Result import ResultPoint, ResultStorage


def train(model, loaders: list, batch_size: int, epochs: int):
    "Train the model and return the loss values."
    # loaders
    train_loader, valid_loader, test_loader = loaders
    # train
    # set model as training mode
    model.train()
    avg_train_losses = []
    avg_valid_losses = []

    for epoch in range(epochs):
        # Losses for each epoch
        train_losses = []
        valid_losses = []

        for X, y in train_loader:
            # Distribute data to device
            X, y = X.to(device), y.to(device).view(-1)
            optimizer.zero_grad()
            output = model(X)
            loss = criterion(output, y)
            train_losses.append(loss.item())
            loss.backward()
            optimizer.step()

        # Validate the model
        accuracy = 0
        model.eval()
        for X, y in valid_loader:
            # Distribute data to device
            X, y = X.to(device), y.to(device).view(
                -1,
            )
            output = model(X)
            loss = criterion(output, y)
            valid_losses.append(loss.item())  # sum up batch loss

            top_p, top_class = output.topk(1, dim=1)
            # See how many of the classes were correct
            equals = top_class == y.view(*top_class.shape)
            # Calculate the mean (get the accuracy for this batch)
            # and add it to the running accuracy for this epoch
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

        # print training/validation statistics
        # calculate average loss over an epoch
        train_loss = np.average(train_losses)
        valid_loss = np.average(valid_losses)
        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)

        accuracy /= len(valid_loader)
        epoch_len = len(str(epochs))

        print_msg = (
            f"[{epoch+1:>{epoch_len}}/{epochs:>{epoch_len}}] "
            + f"train_loss: {train_loss:.5f} "
            + f"valid_loss: {valid_loss:.5f} "
            + f"accuracy: {accuracy:.5f}"
        )
        # Print out the information
        print(print_msg)

    return model, avg_train_losses, avg_valid_losses


def test_model(test_loader, resultStorage: ResultStorage):
    "Run the trained model on test dataset and save output in ResultStorage"
    # Evaluation on test dataset
    model.eval()  # change to evaulation mode
    with torch.no_grad():  # Tell torch not to calculate gradients
        # Iterate over batches in test_laoder
        for X, y in test_loader:
            # Move to device
            X, y = X.to(device), y.to(device).view(-1)
            # Forward pass
            output = model(X)
            # Get the top class of the output
            top_p, top_class = output.topk(1, dim=1)

            # Read images back to PIL
            imgs_batch = []
            # One batch consists of number of sequences (imgs_batch -> imgs -> img)
            # Iterate over points/inputs in a batch
            for point in X.to("cpu"):
                imgs = []
                # Iterate over images in a sequence
                for img in point[0]:  # point[0] as its 1 channel only
                    imgs.append(transforms.ToPILImage()(img))
                imgs_batch.append(imgs)

            # Save the predictions and labels(y)
            preds = top_class.detach().to("cpu").tolist()
            labels = y.view(*top_class.shape).detach().to("cpu").tolist()

            # Iterate over all input-output pairs
            for point in range(len(labels)):
                resultPoint = ResultPoint(
                    imgs_batch[point], preds[point], labels[point]
                )
                resultStorage.add_point(resultPoint)


# %%

"""""""""""""""""""""SET_PARAMS"""""""""""""""""""""""""""""""" 
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# 3D CNN layers and dropout parameters
fc_hidden1, fc_hidden2 = 128, 128
dropout = 0

# training parameters (k = no. target category)
k = 2
img_x, img_y = 320, 240
use_cuda = torch.cuda.is_available()  # check if GPU exists
device = torch.device("cuda" if use_cuda else "cpu")  # use CPU or GPU

# Learning params
epochs = 10
batch_size = 4
learning_rate = 1e-3

# First 10 frames/images selected in each sequence
selected_frames = np.arange(10)
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

# Data directory
data_path = os.path.dirname(os.path.abspath(__file__)) + "/img_slippery"

# Create CNN3D model
model = CNN3D(
    t_dim=len(selected_frames),
    img_x=img_x,
    img_y=img_y,
    drop_p=dropout,
    fc_hidden1=fc_hidden1,
    fc_hidden2=fc_hidden2,
    num_classes=k,
).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# data_loaders = (train, valid, test)
data_loaders = create_loaders(data_path, selected_frames, [img_x, img_y], batch_size)

# TRAIN model
model, train_loss, valid_loss = train(model, data_loaders, batch_size, epochs)

# TEST model
# Create input/output storage object
resultStorage = ResultStorage()
# Run model on the test data
test_model(data_loaders[2], resultStorage)

# Evaluate the model
pred_list = resultStorage.pred_list()
label_list = resultStorage.label_list()
accuracy = resultStorage.compute_accuracy()
print("Accuracy on test data: ", accuracy)

# %% PLOTS

# Visualize the training: loss over epochs
fig = plt.figure(figsize=(10, 8))
plt.plot(range(1, len(train_loss) + 1), train_loss, label="Training Loss")
plt.plot(range(1, len(valid_loss) + 1), valid_loss, label="Validation Loss")

# find position of lowest validation loss
minposs = valid_loss.index(min(valid_loss)) + 1
plt.axvline(minposs, linestyle="--", color="r", label="lowest valid loss")

plt.xlabel("epochs")
plt.ylabel("loss")
plt.ylim(0, 1)  # consistent scale
plt.xlim(0, len(train_loss) + 1)  # consistent scale
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# Plot confusion matrix
labels = [name for name in os.listdir(data_path)
    if os.path.isdir(os.path.join(data_path, name))
]
class_labels = sorted(labels)

cm = sklearn.metrics.confusion_matrix(label_list, pred_list)
disp = sklearn.metrics.ConfusionMatrixDisplay(
    confusion_matrix=cm, display_labels=class_labels
)
disp.plot()
plt.show()

# %%

# Access desired datapoint
print("Size of resultStorage: ", len(resultStorage))
point_num = 33
point = resultStorage.result_list[point_num]
# 0 = not_slipped, 1 = slipped
print("Datapoint ", point_num)
print("prediction: ", point.pred)
print("label: ", point.label)
for i in point.imgs:
    plt.figure()
    plt.imshow(i)

# %%

# RUN model multiple times and determine accuracies
accuracies = []
for i in range(20):
    model = CNN3D(
        t_dim=len(selected_frames),
        img_x=img_x,
        img_y=img_y,
        drop_p=dropout,
        fc_hidden1=fc_hidden1,
        fc_hidden2=fc_hidden2,
        num_classes=k,
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # Create train, valid, test loaders
    data_loaders = create_loaders(data_path, selected_frames)
    model, train_loss, valid_loss = train(model, data_loaders, batch_size, epochs)
    # Test
    test_model(data_loaders[2])
    accuracies.append(resultStorage.compute_accuracy())
