import io

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import CIFAR10
from torchvision.transforms import v2

_CIFAR10_MEAN = [0.4914, 0.4822, 0.4465]
_CIFAR10_STD = [0.2470, 0.2435, 0.2616]


def get_data_loader_CIFAR10C_generated(path, batch_size):
    """Return a DataLoader for the generated CIFAR-10-C parquet dataset."""
    transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=_CIFAR10_MEAN, std=_CIFAR10_STD),
    ])
    dataset = CIFAR10CGenerated(path, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def get_data_loader_CIFAR10(batch_size, train=True, notebook=False, training=True):
    """Return a DataLoader for CIFAR-10 with training or evaluation transforms."""
    path = "../datasets/" if notebook else "datasets/"
    if training and train:
        transform = v2.Compose([
            v2.ToImage(),
            v2.RandomCrop(32, padding=4),
            v2.RandomHorizontalFlip(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=_CIFAR10_MEAN, std=_CIFAR10_STD),
        ])
    else:
        transform = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=_CIFAR10_MEAN, std=_CIFAR10_STD),
        ])

    dataset = CIFAR10(path, download=True, train=train, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


class CIFAR10CGenerated(Dataset):
    """CIFAR-10-C dataset loaded from a pre-generated parquet file."""

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
