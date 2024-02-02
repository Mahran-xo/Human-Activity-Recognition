import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import pandas as pd


class HARDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = pd.read_csv(df)
        self.transform = transform

        # Create a mapping from class names to numerical indices
        self.class_to_idx = {'texting': 0,
                                'sitting': 1,
                                'sleeping': 2,
                                'fighting': 3,
                                'running': 4,
                                'calling': 5,
                                'dancing': 6,
                                'using_laptop': 7,
                                'laughing': 8,
                                'cycling': 9,
                                'hugging': 10,
                                'listening_to_music': 11,
                                'clapping': 12,
                                'eating': 13,
                                'drinking': 14}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        img_path = os.path.join(self.df.iloc[index, 1])
        label_str = self.df.iloc[index, 2]
        label_num = self.class_to_idx[label_str]  # Convert string label to numerical value
        image = np.array(Image.open(img_path).convert("RGB"))
        
        # Apply transformations if specified
        if self.transform:
            augmentations = self.transform(image=image)
            image = augmentations['image']
        
        return {
            'image': image,
            'label': label_num
        }
