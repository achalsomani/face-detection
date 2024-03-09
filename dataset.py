import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder
from torchvision.transforms import transforms

class CustomDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        
        self.label_encoder = LabelEncoder()
        self.data['label_numeric'] = self.label_encoder.fit_transform(self.data['label'])
        self.label_mapping = dict(zip(self.data['label_numeric'], self.data['label']))
        self.num_classes = len(self.data['label_numeric'].unique())

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.data.iloc[idx, 0])
        image = Image.open(img_name)
        label_numeric = self.data.iloc[idx, 2]
        transformed_image = self.transform(image)
        return transformed_image, label_numeric
    
    def numeric_to_string_label(self, numeric_label):
        return self.label_mapping[numeric_label]

