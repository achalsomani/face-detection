import torch
from train import train_model
from utils import get_data_loaders
from model import SimpleCNN
from transforms import train_transforms, test_transforms
import torch.nn as nn
import torch.optim as optim
from params import INPUT_SHAPE, device, LEARNING_RATE, NUM_EPOCHS



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    # Get data loaders
    train_loader, test_loader = get_data_loaders(train_transforms, test_transforms)

    # Instantiate the model
    model = SimpleCNN(input_shape=INPUT_SHAPE, num_classes=train_loader.dataset.num_classes).to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Training loop
    train_model(model, train_loader, test_loader, criterion, optimizer, NUM_EPOCHS)

if __name__ == "__main__":
    main()
