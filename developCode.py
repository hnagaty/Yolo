#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 24 08:35:37 2022

@author: hnagaty
"""


import yoloHelpers as yolo
from torchvision.datasets import VOCDetection
from torch.utils.data import DataLoader
from torchvision import transforms
import torch

S = 7 # An image is divided into SxS grid cells
B = 2 # number of boxes per grid cell
C = 20 # number of classes

data_folder = "/vol/data/dataLocal/PyTorchData" # for local


img_transform = transforms.Compose(
    [transforms.Resize([448, 448]),
     transforms.ToTensor(),
     transforms.Normalize([0.485, 0.456, 0.406],
                          [0.229, 0.224, 0.225])])

label_transform = transforms.Compose(
    [transforms.Lambda(lambda y: yolo.format_label(y)),
     transforms.Lambda(lambda y: torch.tensor(y))])

train_set = VOCDetection(root=data_folder, year="2012",
                         image_set="train", 
                         transform=img_transform,
                         target_transform=label_transform,
                         download=False)

train_set2 = VOCDetection(root=data_folder, year="2012",
                         image_set="train", 
                         # transform=img_transform,
                         # target_transform=label_transform,
                         download=False)
   
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

# For testing
img, annotation = train_set[2]
img
img.shape
img = img.to(device).unsqueeze(0)
annotation.shape 
gt = annotation.to(device).unsqueeze(0)
# lna1 = hp.format_label(annotation)

model = yolo.Yolo().to(device)
print(model)
# confirm that model is on cuda
next(model.parameters()).is_cuda

# forwrad pass a single image
model.eval() # disable batch norm & dropout
# confirm that model is not in training mode
model.training

pred = model(img)

pred1 = pred[0].detach().cpu().numpy()


# Check loss
criterion = yolo.YoloLoss()
loss = criterion(pred, gt)
loss


# empty unused cuda memory
torch.cuda.empty_cache() 


# reset the weights
for layer in model.children():
    print(layer)
    if hasattr(layer, 'reset_parameters'):
        print("Resetting layer")
        layer.reset_parameters()
