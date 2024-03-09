import torch
from utils import calculate_accuracy
from params import device


def train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs):
    for epoch in range(num_epochs):
        train_running_loss = 0.0
        test_running_loss = 0.0

        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_running_loss += loss.item()
        
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
        print(f"Epoch {epoch+1}: Train Loss - {train_loss:.4f}, Test Loss - {test_loss:.4f},\
               Train Accuracy - {100*train_accuracy:.1f}, Test Accuracy - {100*test_accuracy:.1f}")
