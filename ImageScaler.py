import torch
import ssl
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import numpy as np


torch.set_num_threads(1)  # Makes the network only run on a single thread, complying with the requirements
ssl._create_default_https_context = ssl._create_unverified_context  # Turns off the SSL as it was not allowing me to access the dataset

# Define transformation to preprocess the data
transform = transforms.Compose([
    transforms.Resize((64, 64)),  # Resize the image to 64x64
    transforms.Grayscale(),  # Convert the image to grayscale
    transforms.ToTensor(),  # Convert the image to a PyTorch tensor
    #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize the image
    transforms.Normalize(mean=[0.485], std=[0.229])  # Normalize the image for grayscale
])

# Load the Flowers102 dataset as training and test
flowers_train_dataset = datasets.Flowers102(
    root='flowers',    # Directory to save or download the dataset
    transform=transform,
    split='train',
    download=True)

flowers_test_dataset = datasets.Flowers102(
    root='flowers',
    transform=transform,
    split='test',
    download=True)

# Create a DataLoader to iterate over the dataset
# DataLoader allows for easy iteration over the dataset in batches

train_dataloader = DataLoader(flowers_train_dataset, batch_size=64, shuffle=True)
test_dataloader = DataLoader(flowers_test_dataset, batch_size=64, shuffle=True)

# Iterate over the test DataLoader
for images, labels in test_dataloader:
    # Iterate over each image in the batch
    for image in images:
        # Convert the PyTorch tensor to a numpy array and transpose the dimensions
        # This is necessary because Matplotlib expects the channel dimension to be the last dimension
        #image_np = np.transpose(image.numpy(), (1, 2, 0)) #Colour
        image_np = np.squeeze(image.numpy()) #Grey
        # Display the image using Matplotlib
        #plt.imshow(image_np)
        plt.imshow(image_np, cmap='gray')  # Specify the colormap as grayscale
        plt.show()

        # Wait for user input before displaying the next image
        input("Press Enter to continue...")
