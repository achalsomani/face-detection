from torchvision.transforms import transforms
from params import STARTING_SHAPE, INPUT_SHAPE


# Define train transforms
train_transforms = transforms.Compose([
    transforms.Resize(STARTING_SHAPE),  # Resize to 256x256
    transforms.RandomResizedCrop(INPUT_SHAPE, scale=(0.8, 1.0), ratio=(0.8, 1.2)),  # Random crop to 224x224
    transforms.ColorJitter(brightness=0.5),  # Randomly adjust brightness
    transforms.RandomRotation(degrees=20),  # Random rotation
    transforms.RandomHorizontalFlip(p=0.5),  # Random horizontal flip
    transforms.GaussianBlur(kernel_size=3),  # Apply Gaussian blur
    transforms.ToTensor(),  # Convert to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize
])

# Define test transforms
test_transforms = transforms.Compose([
    transforms.Resize(INPUT_SHAPE),  # Resize to 256x256
    transforms.ToTensor(),  # Convert to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize
])
