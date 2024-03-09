from torchvision.transforms import transforms
from params import STARTING_SHAPE, INPUT_SHAPE

# Define train transforms
train_transforms = transforms.Compose([
    transforms.Resize(STARTING_SHAPE),
    transforms.RandomResizedCrop(INPUT_SHAPE, scale=(0.8, 1.0), ratio=(0.8, 1.2)),
    transforms.ColorJitter(brightness=0.5),
    transforms.RandomRotation(degrees=20),
    transforms.RandomHorizontalFlip(p=0.5),
    #transforms.RandomVerticalFlip(p=0.5),
    #transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10),
    #transforms.RandomPerspective(),
    transforms.GaussianBlur(kernel_size=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0, 0, 0], std=[0.5, 0.5, 0.5]),
])

# Define test transforms
test_transforms = transforms.Compose([
    transforms.Resize(INPUT_SHAPE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0, 0, 0], std=[0.5, 0.5, 0.5]),
])
