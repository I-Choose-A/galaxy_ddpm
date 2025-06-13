import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class SDSS(Dataset):
    def __init__(self, transform=None):
        # self.image_path = r"/home/ucaphey/Scratch/sdss.npz"
        # self.properties_path = r"/home/ucaphey/Scratch/sdss_selected_properties.csv"
        self.image_path = r"C:\Users\asus\Desktop\Files\学\UCL\Research Project\Datasets\sdss_slice.npz"
        self.properties_path = r"C:\Users\asus\Desktop\Files\学\UCL\Research Project\Datasets\sdss_selected_properties.csv"
        self.transform = transform

        with np.load(self.image_path, mmap_mode='r') as sdss:
            self.images = sdss["cube"]
        self.properties_df = pd.read_csv(self.properties_path)

    def __len__(self):
        return len(self.properties_df)

    def __getitem__(self, item):
        imageID = self.properties_df.iloc[item, 0]
        image = self.images[imageID]
        properties = self.properties_df.iloc[item, 1:].to_numpy()

        # turn to tensor
        image = torch.from_numpy(image)
        properties = torch.from_numpy(properties)

        # use transform
        if self.transform:
            image = self.transform(image)

        return image, properties
