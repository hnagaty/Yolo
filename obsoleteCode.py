#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 25 08:31:29 2022

@author: hnagaty
"""

def img_transform_f(img):
    tf = transforms.Compose([
            transforms.Resize([448, 448]),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])])        
    img = tf(img)
    return img

