
import torch
import torch.nn as nn  # All neural network modules, nn.Linear, nn.Conv2d, BatchNorm, Loss functions
import torch.optim as optim  # For all Optimization algorithms, SGD, Adam, etc.
import torch.nn.functional as F
from torch.utils.data import (
    DataLoader,
    Dataset,
)  # Gives easier dataset managment and creates mini batches
import torchvision.transforms as transforms  # Transformations we can perform on our dataset
from torchvision.utils import save_image
import pandas as pd
import os
from PIL import Image
from pathlib import Path
from paths import DATA_PATH, CSV_FILE
from sklearn.preprocessing import LabelEncoder

STARTING_SHAPE = (250, 250)
INPUT_SHAPE = (224, 224)
BATCH_SIZE = 16
my_transforms = transforms.Compose(
    [
        transforms.Resize(STARTING_SHAPE), 
        transforms.RandomCrop(INPUT_SHAPE), 
        transforms.ColorJitter(brightness=0.5), 
        transforms.RandomRotation(degrees=20),  
        transforms.RandomHorizontalFlip(p=0.5),  
        transforms.ToTensor(),  # Finally converts PIL image to tensor so we can train w. pytorch
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
        ),
    ]
)

class CustomDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        
        # Use LabelEncoder to convert labels to numeric values
        self.label_encoder = LabelEncoder()
        self.data['label_numeric'] = self.label_encoder.fit_transform(self.data['label'])
        
        # Store the mapping from numeric labels to string labels
        self.label_mapping = dict(zip(self.data['label_numeric'], self.data['label']))
        
        self.num_classes = len(self.data['label_numeric'].unique())

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_name = self.root_dir / self.data.iloc[idx, 0]
        image = Image.open(img_name)
        label_numeric = self.data.iloc[idx, 2]  # Use the numeric label
        transformed_image = self.transform(image)
        return transformed_image, label_numeric
    
    def numeric_to_string_label(self, numeric_label):
        """Convert numeric label to string label"""
        return self.label_mapping[numeric_label]
    
# Load the image dataset
custom_dataset = CustomDataset(csv_file=CSV_FILE, root_dir=DATA_PATH, transform=my_transforms)
# Create a data loader
data_loader = DataLoader(custom_dataset, batch_size=BATCH_SIZE, shuffle=True)

class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        # Adjust the input size to match the new shape
        self.fc1 = nn.Linear(32 * (INPUT_SHAPE[0]//4) * (INPUT_SHAPE[1]//4), 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        # Calculate the size of the flattened features dynamically
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Instantiate the model
model = SimpleCNN(num_classes=custom_dataset.num_classes)  # Update num_classes with your number of classes

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    running_loss = 0.0
    for images, labels in data_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {running_loss/len(data_loader)}")