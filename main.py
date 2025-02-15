import cv2
import numpy as np
import torch
from torch import nn
import torchvision
import torchvision.transforms as transforms
import time
from timeit import default_timer as timer 
from PIL import Image
from ultralytics import YOLO

# TinyVGG model for melanoma detection
class TinyVGG(nn.Module):
    """
    Model architecture copying TinyVGG from: 
    https://poloclub.github.io/cnn-explainer/
    """
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int) -> None:
        super().__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape, 
                      out_channels=hidden_units, 
                      kernel_size=3, # how big is the square that's going over the image
                      stride=1, # default
                      padding=1), # options = "valid" (no padding) or "same" (output has same shape as input) or int for specific number 
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units, 
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,
                         stride=2) # default stride value is same as kernel_size
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_units*16*16,
                      out_features=output_shape)
        )
        
    def forward(self, x: torch.Tensor):
            
        return self.classifier(self.conv_block_2(self.conv_block_1(x))) # <- leverage the benefits of operator fusion

# Function to move servos based on error in x and y coordinates
def move_servos(x_error, y_error, tolerance):
    '''
    returns the amount to change duty cycle of servos based on the error in x and y coordinates of the center of the mole detection box
    '''
    delta_x = 0
    delta_y = 0
    
    step = 0.5
    
    if abs(x_error)>tolerance:
        
        if x_error>0:
            delta_x+=step
        else:
            delta_x-=step
        
    if abs(y_error)>tolerance:
        
        if y_error>0:
            delta_y+=step
        else:
            delta_y-=step
            
    return delta_x, delta_y

# Function to save and return the cropped frame with the mole 
def save_cropped_frame(frame, x, y, img_size=64):
    '''
    saves the frame cropped to the center of the mole with standard image size of 64x64
    '''
    crop = frame[y-img_size//2:y+img_size//2, x-img_size//2:x+img_size//2]
    cv2.imwrite("mole.jpg", crop)
    
#device agnostic code to ensure usage of cuda if avaliable, speeding up runtime and mitigating errors
device = "cuda" if torch.cuda.is_available() else "cpu"

# Both servos start at 90 degrees (center), and constants are set according to servo rotation directions
duty_pan = 7
pan_constant = 1
duty_tilt = 7
tilt_constant = 1

# To capture video from webcam
cap = cv2.VideoCapture(0)

# Center coordinates for screen
screen_center_x = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) / 2)
screen_center_y = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) / 2)

# Range of acceptance for rectangle being center (50 pixels)
tolerance = 50 

# Load the mole detecting model
model = YOLO("MoleDetectionYOLO.pt")

start = 0
end = 0

# Progress indicators on lcd screen
lcd_txt = "Melanoma Detection!"
time.sleep(2)
lcd_txt = "Searching for mole..."

cond = True

while cond:
    
    # Read the frame
    _, frame = cap.read()

    # Detect objects
    results = model(frame)
    
    # Extract bounding boxes and confidence scores
    bboxes = results[0].boxes.xyxy
    confidences = results[0].boxes.conf

    if len(bboxes) > 0:
        # Find index of the highest confidence score
        max_conf_index = confidences.argmax().item()
        
        # Get the bounding box with highest confidence
        x1, y1, x2, y2 = map(int, bboxes[max_conf_index])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

        # Center coordinates for rectangle
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2

        x_error = center_x - screen_center_x
        y_error = center_y - screen_center_y

        # Update start and end times so that time the rectangle is centered can be calculated continuously
        if abs(x_error) <= tolerance and abs(y_error) <= tolerance:
            if start == 0:
                start = timer()   # Begin counting time centered
                
            end = timer()   # Get current time each frame that is centered
            time_elapsed = end - start   # Time elapsed since centered
            
            # Once time elapsed reaches 2 seconds 
            if time_elapsed >= 2:
                print("Mole captured!")
                # Break out of loop
                cond = False
                #save img of mole
                save_cropped_frame(frame, center_x, center_y)
                
        else:
            start = 0   # Reset start timer to 0 once rectangle no longer centered
            
            # Servo error correcting
            delta_x, delta_y = move_servos(x_error, y_error, tolerance)
            print(delta_x, delta_y)
            
            duty_pan += delta_x
            duty_tilt += delta_y
            
            # If any duty cycle falls out of range of servo (2-12), set to center position
            if (duty_pan > 12 or duty_pan < 2) or (duty_tilt > 12 or duty_tilt < 2):
                duty_pan = 7
                duty_tilt = 7

    # Stop if q is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
    
    # Display
    cv2.imshow('Mole Detection', frame)


# Release the VideoCapture object
cap.release()
cv2.destroyAllWindows()

# Progress indicators on lcd screen
lcd_txt = "Mole detected!"
time.sleep(1)
lcd_txt = "Analyzing mole..."

# Load Pytorch Model
melanoma_model = TinyVGG(input_shape = 3,   #3 color channels (RGB)
                        hidden_units = 20,  #from PytorchTraining.py
                        output_shape = 2)     #number of classes in the dataset. 2 for binary classification (benign/melanoma)

# Load the entire checkpoint
checkpoint = torch.load('MelanomaPytorchModel.pth', map_location=torch.device(device))

# Extract the state dictionary (due to how model was saved in previous code)
model_state_dict = checkpoint['model_state_dict']

# Load the state dictionary into the model
melanoma_model.load_state_dict(model_state_dict)
melanoma_model.eval()

#tranform mole image to tensor
transform = transforms.Compose([
        transforms.Resize(64),
        transforms.ToTensor(),
        transforms.Normalize(torch.Tensor([0.7179, 0.5678, 0.5457]), torch.Tensor([0.7179, 0.5678, 0.5457]))
    ])

# Load the mole image and preprocess it
mole_img = Image.open("mole.jpg")
mole_img = transform(mole_img)
mole_img.to(device)
mole_img.unsqueeze_(0)   # Add batch dimension since model expects a batch of images

classes = ["Benign", "Melanoma"]

with torch.inference_mode():

    prediction = melanoma_model(mole_img)
    softmax_values = prediction.softmax(dim=1)
    confidence, predicted_class = torch.max(softmax_values, dim=1)  # Get the class with the highest probability
    
    # Progress indicators on lcd screen, showing the predicted class and confidence percentage
    lcd_txt = f"Predicted class: {classes[predicted_class.item()]}" + f"\nConfidence: {confidence.item() * 100:.2f}%"
    print(lcd_txt)


    