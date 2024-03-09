import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class SimpleCNN(nn.Module):
    def __init__(self, num_classes, pretrained=True, dropout_rate=0.3):
        super(SimpleCNN, self).__init__()
        self.vgg = models.vgg16(pretrained=pretrained)
        self.vgg.eval()  # Set VGG model to evaluation mode by default

        self.fc1 = nn.Linear(512*7*7, 256)  # Adjust input size to match the output size of VGG
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, num_classes)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.vgg.features(x)  # Pass input through VGG
        x = x.view(x.size(0), -1)  # Flatten the output
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


class SimpleCNNOld(nn.Module):
    def __init__(self, input_shape, num_classes, dropout_rate=0.3):
        super(SimpleCNNOld, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)        
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.conv5 = nn.Conv2d(128, 128, kernel_size=3, padding=1)  # New convolutional layer
        self.bn5 = nn.BatchNorm2d(128)                            # New batch normalization layer
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * (input_shape[0] // 32) * (input_shape[1] // 32), 64)
        self.fc2 = nn.Linear(64, num_classes)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        x = self.pool(F.relu(self.bn5(self.conv5(x))))  # Added batch normalization and ReLU after the new convolutional layer
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    
