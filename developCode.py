#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 24 08:35:37 2022

@author: hnagaty
"""


voc_classes = ["background", "aeroplane", "bicycle", "bird", "boat",
               "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
               "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
               "sofa", "train", "tvmonitor"]

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

# Used in training, so I apply the transforms
train_set = VOCDetection(root=data_folder, year="2012",
                         image_set="train", 
                         transform=img_transform,
                         target_transform=label_transform,
                         download=False)

# Used for drawing, so I don't apply the transforms
train_set2 = VOCDetection(root=data_folder, year="2012",
                         image_set="train", 
                         transform=transforms.PILToTensor(),
                         # target_transform=label_transform,
                         download=False)
   
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

# For testing
img, annotation = train_set2[2]
img
img.shape
img = img.to(device).unsqueeze(0)
annotation.shape 
ann = annotation.detach().cpu().numpy()
gt = annotation.to(device).unsqueeze(0)
# lna1 = hp.format_label(annotation)

model = yolo.Yolo().to(device)
print(model)
# confirm that model is on cuda
next(model.parameters()).is_cuda

# load model weights
model.load_state_dict(torch.load('yoloWeights.pth'))

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

#%%        
import numpy as np

S = 7
B = 2
C = 20

conf_threshold = 0.5


boxes = [[1, 0.1, 0.8, 0.5, 0.5],
         [2, 0.7, 0.3, 0.4, 0.4],
         [3, 0.6, 0.6, 0.4, 0.4],
         [4, 0.95, 0.95, 0.6, 0.6]]

label_matrix = np.zeros([S, S, 5 * B + C])
for box in boxes:
    cls, x1, y1, w , h,  = box
    loc = [S * x1, S * y1]
    loc_i = int(loc[1])
    loc_j = int(loc[0])
    y1 = loc[1] - loc_i
    x1 = loc[0] - loc_j
    
    if label_matrix[loc_i, loc_j, C] == 0:
        label_matrix[loc_i, loc_j, C] = 1 # confidence
        label_matrix[loc_i, loc_j, cls] = 1
        label_matrix[loc_i, loc_j, C+1:C+5] = x1, y1, h , w
label_matrix

pred = label_matrix

pred[3, 3, 20]
# put another sample box with other data
pred[3, 3, 20] = 0.7
pred[3, 3, 25] = 0.6
pred[3, 3, 26:30] = 0.9, 0.9, 0.2, 0.2



#%%

gt = yolo.format_label(annotation)
img_x = int(annotation['annotation']['size']['width'])
img_y = int(annotation['annotation']['size']['height'])
p = yolo.pred_to_box(gt, img_x = img_x, img_y = img_y, label_decode=True)

from torchvision.utils import draw_bounding_boxes
import torchvision.transforms.functional as F

def plot_detection(imgT, pred):
    """
    Plots detected objects

    Parameters
    ----------
    img : Image tensor, uint8, C x H x W
        The image tensor    
        This could be obtained by using F.pil_to_tensor(img)
        or by using transforms.PILToTensor()
        F is torchvision.transforms.functional
        
    pred : dict
        Predictions
        Should have those keys
        "boxes": a list of list that has bounding boxes (xmin, ymin, xmax, ymax)
        "labels": a list of the labels

    Returns
    -------
    None.
    """
    return None
    

    
result = draw_bounding_boxes(F.pil_to_tensor(img), 
                             boxes = torch.tensor(pred['boxes']),
                             labels = pred['labels'])
yolo.show(result)

pred = p
