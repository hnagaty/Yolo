# Implemeting YOLOv1 with PyTorch
In this project, we implement YOLOv1 from scratch using PyTorch.  

## What is YOLO
You Look Only Once is an effecticve real-time object detection algorithm first described in 2015 by Joseph Redmon et al.
YOLOv1 is the first incarnation of YOLO. Other version added incremental improvements and enhancements. 

### What files should I look at?
1. `yoloHelpers.py`: Includes all helper functions and the YOLO model class. This is a backbone script that is called by other scripts.
1. `train.py`: The training routine is included in this Python script.
1. `inference.ipynb`: The inference is done in this Jupyter notebook.
1. `lossHistory_20220625.csv`: Just a csv file for the training loss observed during the training.


### Just ignore those files
1. `developCode.py`: This is just a scratch pad that I use for developing new code.
1. `tryCode.py`: This seems like yet another scratch pad.
1. `obsoleteCode.py`: A place holder for functions or code snippets that are no longer used.
1. `downloadData.py`: Just a starter code used for downloading the PASCAL data. It is redundant code and no longer needed.


## So how did it go?
The prediction on training data is acceptable. The model can easily identify objects and define the bounding box.  
For testing data, the model didn't generalise well and suffered from over fitting.  
![trainSample](./figs/trainSample.jpg "Sample Train Images")  
![testSample](./figs/testSample.jpg "Sample Test Images")

