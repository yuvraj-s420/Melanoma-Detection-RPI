# Melanoma-Detection-RPI
Tracks a mole in camera view, moving pan and tilt servos to center the bounding box on screen, and runs PyTorch model to see whether or not it is Melanoma. Additionally shows progress indicators on lcd screen connected to raspberry pi

## Device
![alt text](https://github.com/yuvraj-s420/Melanoma-Detection-RPI/blob/main/Bot_Image.jpg "")

## Model Training Steps

### Dataset Selection

Chose a subset of the ISIC dataset for melanoma and non-melanoma skin lesion images from Kaggle

### Data Annotation

Used CVAT (Computer Vision Annotation Tool) to annotate the dataset for training the PyTorch model

### Data Preparation

Split the dataset into training and testing batches.

Applied preprocessing and normalization for consistent input to the model.

### Model Training

Trained a TinyVGG model on the processed dataset using PyTorch.

Generated training and testing loss/accuracy plots with MatPlotLib to evaluate performance.

### Hyperparameter Optimization

Tuned learning rate, batch size, and number of epochs.

Achieved a maximum accuracy of 93% on the test set.

### Exporting the Model

Saved the trained model as a state dictionary.

Downloaded it onto the Raspberry Pi for deployment.

### Integration with YOLOv8

Used YOLOv8 for initial mole/ blob detection

Passed detected mole images into the trained TinyVGG model for melanoma classification.


## Technologies Used
OpenCV (Yolov8)

PyTorch (TinyVGG model)

MatPlotLib (Displaying loss and accuracy data)

Numpy

Raspberry Pi 5

mg90s Servos (pan and tilt)

Programmable 16x2 LCD Screen
