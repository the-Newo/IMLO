import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

# Define the transformations without normalization
transform_without_normalization = transforms.Compose([
    transforms.RandomResizedCrop(28),
    transforms.Grayscale(),
    transforms.ToTensor(),
])

# Load the training dataset without normalization
training_data = datasets.Flowers102(
    root='flowers',
    transform=transform_without_normalization,
    split='train',
    download=True
)

# Calculate mean and standard deviation of the training dataset
train_mean = torch.stack([img.mean(1).mean(1) for img, _ in training_data]).mean(0)
train_std = torch.stack([img.std(1).std(1) for img, _ in training_data]).std(0)

# Define the transformations with normalization using calculated mean and standard deviation
transform_with_normalization = transforms.Compose([
    transforms.RandomResizedCrop(28),
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize(mean=train_mean, std=train_std)
])

# Load both training and testing datasets with normalization
training_data_normalized = datasets.Flowers102(
    root='flowers',
    transform=transform_with_normalization,
    split='train',
    download=True
)

test_data_normalized = datasets.Flowers102(
    root='flowers',
    transform=transform_with_normalization,
    split='test',
    download=True
)

train_dataloader = DataLoader(training_data_normalized, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data_normalized, batch_size=64)

# Simplified model architecture
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 256),  # Reduced the number of units
            nn.ReLU(),
            nn.Linear(256, 102),  # Output size is kept same as the number of classes
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork()

# Decreased learning rate
learning_rate = 1e-4
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        pred = model(X)
        loss = loss_fn(pred, y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Print the loss for every batch
        loss, current = loss.item(), batch * len(X) + len(X)
        print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")




def test_loop(dataloader, model, loss_fn):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

epochs = 10
for t in range(epochs):
    print(f"Epoch {t + 1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)
print("Done!")

