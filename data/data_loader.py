import io

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import CIFAR10
from torchvision.transforms import v2

from data.data_utils import load_data


def get_data_loader_CIFAR10C(batch_size):
    return DataLoader(CIFAR10C(), batch_size=batch_size, shuffle=True)


def get_data_loader_CIFAR10C_generated(path, batch_size):
    transform = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])

    dataset = CIFAR10CGenerated(path, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def get_data_loader_CIFAR10(batch_size, train=True, notebook=False, training=True):
    path = "../datasets/" if notebook else "datasets/"
    transform = (
        v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
        if training
        else None
    )

    dataset = CIFAR10(path, download=True, train=train, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


class CIFAR10CGenerated(Dataset):
    def __init__(self, path, transform=None):
        df = pd.read_parquet(path)
        self.images = df["image"].tolist()
        self.labels = df["label"].tolist()
        self.corruption_names = df["corruption_name"].tolist()
        self.corruption_levels = df["corruption_level"].tolist()
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        img = Image.open(io.BytesIO(self.images[index]))
        if self.transform:
            img = self.transform(img)
        return (
            img,
            self.labels[index],
            self.corruption_names[index],
            self.corruption_levels[index],
        )


class CIFAR10C(Dataset):
    def __init__(self):
        self.ds = load_data()

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index):
        item = self.ds[index]
        img = item["image"]
        return img, item["label"]
