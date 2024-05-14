import torch
import ssl
import time
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch import nn
import matplotlib.pyplot as plt

#torch.set_num_threads(1)  # Makes the network only run on a single thread, complying with the requirements
ssl._create_default_https_context = ssl._create_unverified_context  # Turns off the SSL as it was not allowing me to access the dataset

# Define transformation to preprocess the data
from torchvision import transforms

# Define transformation to preprocess the data with data augmentation
transform = transforms.Compose([
    transforms.RandomRotation(degrees=30),  # Randomly rotate the image by a maximum of 30 degrees
    transforms.RandomHorizontalFlip(),      # Randomly flip the image horizontally
    transforms.RandomResizedCrop(224),      # Randomly crop and resize the image to 224x224
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),  # Randomly adjust brightness, contrast, saturation, and hue
    transforms.ToTensor(),                  # Convert the image to a PyTorch tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])   # Normalize the image
])


# Load the Flowers102 dataset as training and test
# Load the Flowers102 dataset for training and test
flowers_train_dataset = datasets.Flowers102(
    root='flowers',
    transform=transform,
    split='train',
    download=True)

flowers_test_dataset = datasets.Flowers102(
    root='flowers',
    transform=transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
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

class ImprovedNeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(224 * 224 * 3, 2048),
            nn.BatchNorm1d(2048),  # Batch normalization added
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(2048, 2048),
            nn.BatchNorm1d(2048),  # Batch normalization added
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),  # Batch normalization added
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),   # Batch normalization added
            nn.ReLU(),
            nn.Linear(512, 102),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits




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
training_time = 3600 * 10
start_time = time.time()
elapsed_time = 0

while elapsed_time < training_time:
    # Perform training loop and measure accuracy
    train_loop(train_dataloader, improved_model, loss_fn, optimizer)
    # Update elapsed time
    elapsed_time = time.time() - start_time
    # Print elapsed time and current accuracy
    print(f"Elapsed time: {elapsed_time}")

current_accuracy = test_loop(test_dataloader, improved_model, loss_fn)
print(f"Elapsed time: {elapsed_time} seconds, Current accuracy: {current_accuracy:.2f}%")
print("Training time limit reached. Done!")




