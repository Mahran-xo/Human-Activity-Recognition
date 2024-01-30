import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import pandas as pd


class HARDataset(Dataset):
    def __init__(self, dir, df,transform=None):
        self.df = pd.read_csv(df)
        self.dir = dir
        self.image_names = self.df.filename
        self.labels = list(self.df.label)
        self.transform = transform
        self.class_names = [
            'calling',
            'clapping',
            'cycling',
            'dancing',
            'drinking',
            'eating',
            'fighting',
            'hugging',
            'laughing',
            'listening_to_music',
            'running',
            'sitting',
            'sleeping',
            'texting',
            'using_laptop'
        ]

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, index):
        img_path = os.path.join(self.dir, self.image_names[index])
        label = self.labels[index]
        image = np.array(Image.open(img_path).convert("RGB"))
        augmentations = self.transform(image=image)
        image = augmentations['image']
        class_num = self.class_names.index(label)
        return {
            'image': image,
            'label': class_num
        }
