# data_loader.py

"""
Loads the dataset for the model from the HERMES/data/dataset folder, creating
a dataset class.
"""

### IMPORT LIBRARIES :

import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

### DEFINE DATASET CLASS :

class BuildingDataset(Dataset):
    """
    Dataset for associating impact photos to labels (x1,y1,x2,y2).
    Expects a CSV 'labels.csv' in the ml directory, and a 'photos' subfolder with images.
    """

    def __init__(self, ml_dir, transform = None):

        # Paths :
        self.ml_dir = ml_dir
        self.csv_path = os.path.join(ml_dir, '..', 'data', 'dataset', 'labels.csv')
        photos_dir = os.path.join(ml_dir, '..', 'data', 'dataset', 'photos')

        # Load labels :
        df = pd.read_csv(self.csv_path)
        self.images = df['filename'].tolist()
        self.labels = df[['x1', 'y1', 'x2', 'y2']].values.astype('float32')
        self.photos_dir = photos_dir
        self.transform = transform or transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.photos_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        label = torch.tensor(self.labels[idx])
        return image, label
