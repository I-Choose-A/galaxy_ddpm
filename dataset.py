import numpy as np
import pandas as pd
import sep
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

# real data of 1 morphological category
class SDSS_Single_Class(Dataset):
    def __init__(self, images_path, conditions_path, classification, transform=None):
        self.images_path = images_path
        self.conditions_path = conditions_path
        self.transform = transform

        with np.load(self.images_path, mmap_mode='r') as sdss:
            self.images = sdss["cube"]
        conditions_df = pd.read_csv(self.conditions_path)
        sample_num = [4560, 4739, 500, 1935, 10341]
        self.conditions_df = conditions_df[conditions_df["mapped_gz2class"] == classification]
        self.conditions_df = self.conditions_df.reset_index(drop=True).head(sample_num[classification])

    def __len__(self):
        return len(self.conditions_df)

    def __getitem__(self, item):
        imageID = self.conditions_df.loc[item, "imageID"]
        image = self.images[imageID]
        condition = np.array(self.conditions_df.iloc[item, 3:], dtype=np.float32)

        for i in range(5):
            channel = np.ascontiguousarray(image[:, :, i])
            bkg = sep.Background(channel, bw=64, bh=64)
            bkg_mean = bkg.back()
            image[..., i] = channel - bkg_mean

        # use transform
        if self.transform:
            image = self.transform(image)

        return image, condition

# generated data of 1 morphological category
class FakeData(Dataset):
    def __init__(self, images_path, classification, transform=None):
        self.images_path = images_path
        self.transform = transform

        self.images = np.load(self.images_path)
        sample_num = self.images.shape[0]
        self.labels = np.repeat(classification, sample_num)

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, item):
        image = self.images[item]
        label = self.labels[item]

        image = torch.from_numpy(image)
        # use transform
        if self.transform:
            image = self.transform(image)

        return image, label

# generated data of all morphological categories
class FakeData_for_IS(Dataset):
    def __init__(self, images_path, transform=None):
        self.images_path = images_path
        self.transform = transform

        self.images = np.load(self.images_path)
        sample_num = [1000] * 5
        self.labels = np.repeat([0, 1, 2, 3, 4], sample_num)

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, item):
        image = self.images[item]
        label = self.labels[item]

        image = torch.from_numpy(image)
        # use transform
        if self.transform:
            image = self.transform(image)

        return image, label
