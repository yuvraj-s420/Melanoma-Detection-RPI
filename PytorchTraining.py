#import libraries used
import numpy as np
import torch
from torch import nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.optim.lr_scheduler as lr_scheduler
from timeit import default_timer as timer 
'''
All of these functions are helper functions from another file full of basic function templates from pytorch official documentation
important_functions.py
'''

#device agnostic code to ensure usage of cuda if avaliable, speeding up runtime and mitigating errors
device = "cuda" if torch.cuda.is_available() else "cpu"


# Calculate accuracy (a classification metric)
def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item() # torch.eq() calculates where two tensors are equal
    acc = (correct / len(y_pred)) * 100 
    return acc

#trains model
def train_step(model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               accuracy_fn,
               device: torch.device = device):
    '''
    Trains model using inputted model, data loader, loss function, optimizer, accuracy function, and device

    '''

    train_loss, train_acc = 0, 0
    model.to(device)

    #trains model through each batch
    for batch, (X, y) in enumerate(data_loader):


        # Send data to current device
        X, y = X.to(device), y.to(device)

        # 1. Forward pass
        y_pred = model(X)

        # 2. Calculate loss
        loss = loss_fn(y_pred, y)
        train_loss += loss
        train_acc += accuracy_fn(y_true=y,
                                 y_pred=y_pred.argmax(dim=1)) # Go from logits -> pred labels

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward
        loss.backward()

        # 5. Optimizer step
        optimizer.step()

    # Calculate total loss and accuracy of each epoch and returns it
    train_loss /= len(data_loader)
    train_acc /= len(data_loader)
    return train_loss, train_acc

#tests model
def test_step(model: torch.nn.Module,
              data_loader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              accuracy_fn,
              device: torch.device = device):
    
    test_loss, test_acc = 0, 0
    model.to(device)
    model.eval() # put model in eval mode
    # Turn on inference context manager
    with torch.inference_mode(): 
        for X, y in data_loader:
            # Send data to GPU
            X, y = X.to(device), y.to(device)
            
            # 1. Forward pass
            test_pred = model(X)
            
            # 2. Calculate loss and accuracy
            test_loss += loss_fn(test_pred, y)
            test_acc += accuracy_fn(y_true=y,
                y_pred=test_pred.argmax(dim=1) # Go from logits -> pred labels
            )
        
        # Adjust metrics and print out
        test_loss /= len(data_loader)
        test_acc /= len(data_loader)
        return test_loss, test_acc

# evaluates model
def eval_model(model: torch.nn.Module, 
               data_loader: torch.utils.data.DataLoader, 
               loss_fn: torch.nn.Module, 
               accuracy_fn,
               device: torch.device = device):
    """Returns a dictionary containing the results of model predicting on data_loader.

    Args:
        model (torch.nn.Module): A PyTorch model capable of making predictions on data_loader.
        data_loader (torch.utils.data.DataLoader): The target dataset to predict on.
        loss_fn (torch.nn.Module): The loss function of model.
        accuracy_fn: An accuracy function to compare the models predictions to the truth labels.

    Returns:
        (dict): Results of model making predictions on data_loader.
    """
    loss, acc = 0, 0
    model.eval()
    with torch.inference_mode():
        for X, y in data_loader:
            # Make predictions with the model
            y_pred = model(X)
            
            # Accumulate the loss and accuracy values per batch
            loss += loss_fn(y_pred, y)
            acc += accuracy_fn(y_true=y, 
                                y_pred=y_pred.argmax(dim=1)) # For accuracy, need the prediction labels (logits -> pred_prob -> pred_labels)
        
        # Scale loss and acc to find the average loss/acc per batch
        loss /= len(data_loader)
        acc /= len(data_loader)
        
    return {"model_name": model.__class__.__name__, # only works when model was created with a class
            "model_loss": loss.item(),
            "model_acc": acc}

def save_model(model):
    """
    Function to save the trained model to disk.
    """
    torch.save({'model_state_dict': model.state_dict()
                }, 'outputs/model.pth')

def show_transformed_images(dataset):
    '''
    shows a sample of 6 images from a dataset in matplotlib
    '''
    loader = torch.utils.data.DataLoader(dataset, batch_size = 6, shuffle = True)
    batch = next(iter(loader))
    images, labels = batch

    grid = torchvision.utils.make_grid(images, nrow = 3)
    plt.figure(figsize = (11,11))
    plt.imshow(np.transpose(grid, (1,2,0)))
    print("labels: ", labels)
    plt.show()

def save_plots(train_acc, test_acc, train_loss, test_loss):
    """
    Function to save the loss and accuracy plots to disk.
    inputs are lists
    """
    # accuracy plots
    plt.figure(figsize=(10, 7))
    plt.plot(
        train_acc, color='green', linestyle='-', 
        label='train accuracy'
    )
    plt.plot(
        test_acc, color='blue', linestyle='-', 
        label='test accuracy'
    )
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('outputs/accuracy.png')

    # loss plots
    plt.figure(figsize=(10, 7))
    plt.plot(
        train_loss, color='orange', linestyle='-', 
        label='train loss'
    )
    plt.plot(
        test_loss, color='red', linestyle='-', 
        label='test loss'
    )
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('outputs/loss.png')

# TinyVGG model
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
                      kernel_size=3, # how big is the square that's going over the image?
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
            # Where did this in_features shape come from? 
            # It's because each layer of our network compresses and changes the shape of our input data.
            nn.Linear(in_features=hidden_units*16*16,
                      out_features=output_shape)
        )
    
    def forward(self, x: torch.Tensor):
        # x = self.conv_block_1(x)
        # print(x.shape)
        # x = self.conv_block_2(x)
        # print(x.shape)
        # x = self.classifier(x)
        # print(x.shape)
        # return x
        return self.classifier(self.conv_block_2(self.conv_block_1(x))) # <- leverage the benefits of operator fusion

if __name__ == "__main__":

    # Hyperparameters
    # change and tweak to improve model accuracy
    IMAGE_SIZE = 64    #many neural networks prefer same sized images. size of original images is 300
    BATCH_SIZE = 256     #how many images/datapoints are viewed at each iteration (batches chunk the data to improve efficiency)
    HIDDEN_UNITS = 20   #how many hidden variables/ parameters used to predict patterns in the model
    LEARNING_RATE = 0.01
    EPOCHS = 100

    #set manual seed to ensure reproducability and eliminate randomness allowing for better control of variables
    torch.manual_seed(77)
    torch.cuda.manual_seed(77)

    # Paths to datasets
    # Gotten from Kaggle at: https://www.kaggle.com/datasets/hasnainjaved/melanoma-skin-cancer-dataset-of-10000-images?resource=download
    train_data_path = "melanoma_cancer_dataset/train"
    test_data_path = "melanoma_cancer_dataset/test"

    #mean and std gathered from melanoma_std_mean.py
    mean = [0.7179, 0.5678, 0.5457]
    std = [0.7179, 0.5678, 0.5457]

    start = timer()

    # Transforms done on train data, such as resizing
    # Additional Transforms are done (data augmentation) to artificially increase diversity in dataset
    train_transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),  #converts all pixel values from 0-255 into 0-1
        transforms.Normalize(torch.Tensor(mean), torch.Tensor(std))
    ])

    # Transforms done on test data: resizes, normalization, no additional transforms required as test data is used to validate, not train
    test_transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(torch.Tensor(mean), torch.Tensor(std))
    ])

    # Create training dataset from training folder and apply train transforms
    # ImageFolder method is a convenient way to organize all the picture data from a folder with train, test, and classes within each as a tensor
    # data is of form
    train_dataset = torchvision.datasets.ImageFolder(root = train_data_path, transform = train_transform, target_transform = None)  #do not apply transforms to targets in this case
    #of shape (3,IMAGE_SIZE,IMAGE_SIZE)

    test_dataset = torchvision.datasets.ImageFolder(root = test_data_path, transform = test_transform, target_transform = None)  #do not apply transforms to targets in this case

    #holds dictionary of class names and index
    class_dict = train_dataset.class_to_idx     # {'benign': 0, 'malignant': 1}

    train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                batch_size=BATCH_SIZE, # how many samples per batch?
                                num_workers=0, # how many subprocesses to use for data loading? (higher = more). Use os.cpu_count() to use as many as possible on a device
                                shuffle=True) # shuffle the data?

    test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                batch_size=BATCH_SIZE, 
                                num_workers=0, 
                                shuffle=False) # don't usually need to shuffle testing data

    model_0 = TinyVGG(input_shape = 3,  #3 color channels (RGB)
                        hidden_units = HIDDEN_UNITS, 
                        output_shape = 2)     #number of classes in the dataset. 2 for binary classification (benign/melanoma)

    # Loss Function
    loss_fn = nn.CrossEntropyLoss()

    # Optimizer
    optimizer = torch.optim.Adam(params=model_0.parameters(), lr=LEARNING_RATE)

    # lr scheduler
    scheduler = lr_scheduler.StepLR(optimizer,step_size=10, gamma=0.3)

    end = timer()

    print(f"Time taken to set up model: {end-start:.2f}\n")

    train_acc, test_acc, train_loss, test_loss = [],[],[],[]

    # Training loop
    for epoch in range(1,EPOCHS+1):

        #tracking the time it takes to test/train each epoch
        start = timer()

        train_loss_current, train_acc_current = train_step(model= model_0, 
                                           data_loader= train_dataloader, 
                                           loss_fn= loss_fn, 
                                           optimizer= optimizer, 
                                           accuracy_fn= accuracy_fn, 
                                           device= device)
        
        #train_loss_current is a tensor since it is gotten from loss function, thus needs to be turned into an item for save_plots() to function
        train_loss.append(train_loss_current.item())
        train_acc.append(train_acc_current)
        
        test_loss_current, test_acc_current = test_step(model= model_0, 
                                        data_loader= test_dataloader, 
                                        loss_fn= loss_fn, 
                                        accuracy_fn= accuracy_fn, 
                                        device= device)
        
        test_loss.append(test_loss_current.item())  #convert to value only, not tensor
        test_acc.append(test_acc_current)
        
        end = timer()

        scheduler.step()

        print(f"Epoch {epoch}:\tTime taken: {end-start:.2f}s\nTraining loss: {train_loss_current:.4f}\tTesting loss: {test_loss_current:.4f}\tTraining Accuracy: {train_acc_current:.2f}\tTesting Accuracy: {test_acc_current:.2f}\n")

        #save data to a plot in folder
        save_plots(train_acc, test_acc, train_loss, test_loss)

        #save model to file in /outputs/model.pth
        if epoch%10==0:
            save_model(model_0)


    

    
    
