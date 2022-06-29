#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Sat Jun 25 23:17:33 2022

@author: hnagaty
"""

import yoloHelpers as hp
from torchvision.datasets import VOCDetection
from torch.utils.data import DataLoader
from torchvision import transforms
import torch
import time

S = 7 # An image is divided into SxS grid cells
B = 2 # number of boxes per grid cell
C = 20 # number of classes

batch_size = 16
learning_rate = 1e-4
epochs = 100
weight_decay = 0.0005
momentum = 0.9

data_folder = "/vol/data/dataLocal/PyTorchData" # for local


img_transform = transforms.Compose(
    [transforms.Resize([448, 448]),
     transforms.ToTensor(),
     transforms.Normalize([0.485, 0.456, 0.406],
                          [0.229, 0.224, 0.225])])

label_transform = transforms.Compose(
    [transforms.Lambda(lambda y: hp.format_label(y)),
     transforms.Lambda(lambda y: torch.tensor(y))])

# train data
train_set = VOCDetection(root=data_folder, year="2012",
                         image_set="train", 
                         transform=img_transform,
                         target_transform=label_transform,
                         download=False)
train_loader = DataLoader(train_set,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=2)

# validation data
val_set = VOCDetection(root=data_folder, year="2012",
                         image_set="val", 
                         transform=img_transform,
                         target_transform=label_transform,
                         download=False)
val_loader = DataLoader(val_set,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=2)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

model = hp.Yolo().to(device)
criterion = hp.YoloLoss()
optimizer = torch.optim.Adam(model.parameters(),
                             lr=learning_rate,
                             weight_decay=weight_decay)

def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    loss_history = []
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            loss_history.append(loss)
    return loss_history
        

st = time.time()
all_loss_history = []
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    l = train_loop(train_loader, model, criterion, optimizer)
    all_loss_history.extend(l)
et = time.time()
elapsed_time = et - st
print(f'Execution time: {elapsed_time/(60*60)} hours')

torch.save(model.state_dict(), 'yoloWeights.pth')

