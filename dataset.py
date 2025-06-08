import argparse

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

parser = argparse.ArgumentParser()

parser.add_argument("sdss_path", help="Path of SDSS dataset")
parser.add_argument("properties_path", help="Path of selected properties from SDSS dataset")
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SDSS(Dataset):
    def __init__(self, transform=None):
        self.image_path = args.sdss_path
        self.properties_path = args.properties_path
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

        # use transform）
        if self.transform:
            image = self.transform(image)

        return image, properties
