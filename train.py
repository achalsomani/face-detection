import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from dataset import CustomDataset
from model import SimpleCNN
from paths import DATA_PATH, CSV_FILE, TESTING_DATA_PATH, TESTING_CSV_FILE

STARTING_SHAPE = (225, 225)
INPUT_SHAPE = (200, 200)
BATCH_SIZE = 64

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_transforms = transforms.Compose([
    transforms.Resize(STARTING_SHAPE),
    transforms.RandomCrop(INPUT_SHAPE),
    transforms.ColorJitter(brightness=0.5),
    transforms.RandomRotation(degrees=20),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

test_transforms = transforms.Compose([
    transforms.Resize(INPUT_SHAPE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

def calculate_accuracy(model, data_loader):
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total

def train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs):
    for epoch in range(num_epochs):
        train_running_loss = 0.0
        test_running_loss = 0.0

        # Training
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_running_loss += loss.item()
        
        # Testing
        model.eval()
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                test_running_loss += loss.item()

        # Print epoch statistics
        train_loss = train_running_loss / len(train_loader)
        test_loss = test_running_loss / len(test_loader)
        train_accuracy = calculate_accuracy(model, train_loader)
        test_accuracy = calculate_accuracy(model, test_loader)
        print(f"Epoch {epoch+1}: Train Loss - {train_loss:.4f}, Test Loss - {test_loss:.4f}, Train Accuracy - {train_accuracy:.4f}, Test Accuracy - {test_accuracy:.4f}")

# Load the train and test image datasets
train_dataset = CustomDataset(csv_file=CSV_FILE, root_dir=DATA_PATH, transform=train_transforms)
test_dataset = CustomDataset(csv_file=TESTING_CSV_FILE, root_dir=TESTING_DATA_PATH, transform=test_transforms)

# Create data loaders for train and test datasets
train_data_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_data_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Instantiate the model
model = SimpleCNN(input_shape=INPUT_SHAPE, num_classes=train_dataset.num_classes).to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
learning_rate = 0.001
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
num_epochs = 40
train_model(model, train_data_loader, test_data_loader, criterion, optimizer, num_epochs)
