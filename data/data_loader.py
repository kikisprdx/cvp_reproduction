from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import CIFAR10

from data.data_utils import load_data


def get_data_loader_CIFAR10C(batch_size):
    return DataLoader(CIFAR10C(), batch_size=batch_size, shuffle=True)


def get_data_loader_CIFAR10(batch_size, notebook=False):
    if notebook:
        dataset = CIFAR10("../datasets/", download=True)
    else:
        dataset = CIFAR10("datasets/", download=True)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


class CIFAR10C(Dataset):
    def __init__(self):
        self.ds = load_data()

    def __len__(self):
        return len(self.ds)

    def __get_item__(self, idx):
        item = self.ds[idx]
        img = item["image"]
        return img, item["label"]
