import torch
import ssl
import time
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch import nn
import matplotlib.pyplot as plt

# torch.set_num_threads(1)  # Makes the network only run on a single thread, complying with the requirements
ssl._create_default_https_context = ssl._create_unverified_context  # Turns off the SSL as it was not allowing me to access the dataset

# Define transformation to preprocess the data
# Define transformation to preprocess the data
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),  # Rotate image randomly up to 10 degrees
    transforms.Resize((64, 64)),  # Resize images to the same dimensions
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize images
])

# No augmentation for validation/test set
val_transform = transforms.Compose([
    transforms.Resize((64, 64)),  # Resize images to the same dimensions
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize images
])

# Load the Flowers102 dataset as training and test
flowers_train_dataset = datasets.Flowers102(
    root='flowers',
    transform=train_transform,
    split='train',
    download=True)

flowers_test_dataset = datasets.Flowers102(
    root='flowers',
    transform=val_transform,
    split='test',
    download=True)


# Creates a DataLoader to iterate over each dataset in batches of 64
train_dataloader = DataLoader(flowers_train_dataset, batch_size=64, shuffle=True)
test_dataloader = DataLoader(flowers_test_dataset, batch_size=64, shuffle=True)

# Decides where the program will be run (usually CPU core)
device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"  # Corrected device assignment
)

import torch.nn as nn

class ImprovedNeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_stack = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc_stack = nn.Sequential(
            nn.Linear(512 * 8 * 8, 512),  # Adjusted input size
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 102)
        )

    def forward(self, x):
        x = self.conv_stack(x)
        x = x.view(x.size(0), -1)  # Flatten the output of conv layers
        x = self.fc_stack(x)
        return x



# Create an instance of the improved model
improved_model = ImprovedNeuralNetwork().to(device)

learning_rate = 1e-3
batch_size = 64

def train_loop(dataloader, model, loss_fn, optimizer, scheduler=None, max_iterations=16):
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        optimizer.zero_grad()
        pred = model(X.to(device))
        loss = loss_fn(pred, y.to(device))
        loss.backward()
        optimizer.step()
        if scheduler:
            scheduler.step()  # Step the scheduler

        # Print batch information
        loss_val = loss.item()
        print(f"Batch {batch}/{len(dataloader)}, Loss: {loss_val}")

        if batch == max_iterations - 1:
            break

def test_loop(dataloader, model, loss_fn):
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X.to(device))
            test_loss += loss_fn(pred, y.to(device)).item()
            correct += (pred.argmax(1) == y.to(device)).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    accuracy = correct * 100
    return accuracy

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(improved_model.parameters(), lr=learning_rate, weight_decay=1e-4)  # L2 regularization added

# Learning rate scheduler
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)  # StepLR scheduler

# Initialize variables for training
training_time = 60 * 60 * 7.5

start_time = time.time()
elapsed_time = 0

#AccMark = []


while elapsed_time < training_time:
    # Perform training loop and measure accuracy
    train_loop(train_dataloader, improved_model, loss_fn, optimizer)
    # Update elapsed time
    elapsed_time = time.time() - start_time
    # Print elapsed time and current accuracy
    #print(f"Elapsed time: {elapsed_time}")
    #AccMark.append(test_loop(test_dataloader, improved_model, loss_fn))

current_accuracy = test_loop(test_dataloader, improved_model, loss_fn)
print(f"Elapsed time: {elapsed_time} seconds, Current accuracy: {current_accuracy:.2f}%")
print("Training time limit reached. Done!")

#for i in AccMark:
#    print(str(i))

# Creating the plot
#plt.figure(figsize=(10, 6))
#plt.plot(AccMark, marker='o', linestyle='-')

# Adding labels and title
#plt.xlabel('Data Point')
#plt.ylabel('Value')
#plt.title('Plot of Provided Values')

# Displaying the plot
#plt.grid(True)
#plt.show()

#32% after 7.5h