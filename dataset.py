import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import sep


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
        condition = np.array(self.conditions_df.iloc[item, 3:], dtype=np.float32)

        for i in range(len(self.channels)):
            channel = np.ascontiguousarray(image[:, :, i])
            bkg = sep.Background(channel, bw=64, bh=64)
            bkg_mean = bkg.back()
            image[..., i] = channel - bkg_mean

        # use transform
        if self.transform:
            image = self.transform(image)

        # select needed channels
        image = image[self.channels, :, :]

        return image, condition
