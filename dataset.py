import cv2
import numpy as np
import albumentations
import torch
from torch.utils.data import Dataset
from transformers import *
from tqdm import tqdm

class ShopeeDataset(Dataset):
    def __init__(self, df, path, mode, transform=None):
        self.df = df
        self.path = path
        self.mode = mode
        self.file_names = df['image'].values
        self.transform = transform
        self.labels = df['label_group'].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_name = self.file_names[idx]
        file_path = self.path + file_name
        image = cv2.imread(file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        image = image.astype(np.float32)
        image = image.transpose(2, 0, 1)
        label = torch.tensor(self.labels[idx]).long()

        if self.mode == 'train':
            return torch.tensor(image).float(), label
        elif self.mode == 'valid':
            return torch.tensor(image).float()

def get_transforms(image_size):

    transforms_train = albumentations.Compose([

        albumentations.HorizontalFlip(p=0.5),
        albumentations.ImageCompression(quality_lower=99, quality_upper=100),
        albumentations.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=10, border_mode=0, p=0.7),
        albumentations.Resize(image_size, image_size),
        albumentations.Cutout(max_h_size=int(image_size * 0.4), max_w_size=int(image_size * 0.4), num_holes=1, p=0.5),
        albumentations.Normalize(),

    ])

    transforms_val = albumentations.Compose([
        albumentations.Resize(image_size, image_size),
        albumentations.Normalize()
    ])

    return transforms_train, transforms_val
