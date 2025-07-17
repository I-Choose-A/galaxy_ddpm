import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class SDSS(Dataset):
    def __init__(self, images_path, conditions_path, channels=None, transform=None):
        self.images_path = images_path
        self.conditions_path = conditions_path
        self.transform = transform

        self.original_channels = ["u", "g", "r", "i", "z"]
        if channels:
            self.channels = [self.original_channels.index(ch) for ch in channels]
        else:
            self.channels = [0, 1, 2, 3, 4]

        with np.load(self.images_path, mmap_mode='r') as sdss:
            self.images = sdss["cube"]
        self.conditions_df = pd.read_csv(self.conditions_path)

    def __len__(self):
        return len(self.conditions_df)

    def __getitem__(self, item):
        imageID = self.conditions_df.loc[item, "imageID"]
        image = self.images[imageID]
        condition = self.conditions_df.loc[item, "mapped_gz2class"]

        # turn to tensor
        image = torch.from_numpy(image)

        # use transform
        if self.transform:
            image = self.transform(image)

        # select needed channels
        image = image[self.channels, :, :]

        return image, condition
