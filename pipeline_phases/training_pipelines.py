import os

import torch
from torchvision import transforms

from models.CVP_model import CVPF3, CVPR3
from pipeline_phases.CVP_trainer import CVPTrainer
from pipeline_phases.SVP_trainer import SVPTrainer


def train_svp(size, svp_model, optimiser, train_data, test_loader):
    data_transforms = transforms.Compose(
        [
            transforms.RandomResizedCrop(size=size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=[-90, 90]),
        ]
    )
    return SVPTrainer(svp_model, optimiser, train_data, test_loader, transforms=data_transforms, n_views=3)


def _cvp_transforms():
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(size=32),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=90),
        ]
    )


def train_cvpf3(ssl_model, base_model, train_data, test_loader):
    device = next(base_model.parameters()).device
    cvp_model = CVPF3(base_model, 3, ssl_model)
    cvp_model.to(device)
    optimiser = torch.optim.SGD(cvp_model.head.parameters(), lr=0.001)
    return CVPTrainer(cvp_model, optimiser, training_data=train_data, test_data=test_loader, tau=0.2, transforms=_cvp_transforms(), n_views=3)


def train_cvpr3(ssl_model, base_model, train_data, test_loader):
    device = next(base_model.parameters()).device
    cvp_model = CVPR3(base_model, 3, ssl_model)
    cvp_model.to(device)
    optimiser = torch.optim.SGD(cvp_model.head.parameters(), lr=0.001)
    return CVPTrainer(cvp_model, optimiser, training_data=train_data, test_data=test_loader, tau=0.2, transforms=_cvp_transforms(), n_views=3)
