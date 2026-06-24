import os

import torch
from torchvision import transforms

from src.models.CVP_model import CVPF3, CVPR3
from src.pipeline_phases.CVP_trainer import CVPTrainer
from src.pipeline_phases.SVP_trainer import SVPTrainer


def train_svp(size, svp_model, optimiser, train_data, test_loader):
    """Construct and return an SVPTrainer with standard augmentation transforms."""
    data_transforms = transforms.Compose(
        [
            transforms.RandomResizedCrop(size=size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=[-90, 90]),
        ]
    )
    return SVPTrainer(svp_model, optimiser, train_data, test_loader, transforms=data_transforms, n_views=3)


def _cvp_transforms():
    """Return the augmentation pipeline used for CVP contrastive views."""
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(size=32),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=90),
        ]
    )


def train_cvpf3(ssl_model, base_model, train_data, test_loader):
    """Construct and return a CVPTrainer for the fixed-kernel CVP-F3 variant."""
    device = next(base_model.parameters()).device
    cvp_model = torch.compile(CVPF3(base_model, 3, ssl_model).to(device))
    optimiser = torch.optim.SGD(cvp_model.head.parameters(), lr=0.001)
    return CVPTrainer(cvp_model, optimiser, training_data=train_data, test_data=test_loader, tau=0.2, transforms=_cvp_transforms(), n_views=3)


def train_cvpr3(ssl_model, base_model, train_data, test_loader):
    """Construct and return a CVPTrainer for the random-kernel CVP-R3 variant."""
    device = next(base_model.parameters()).device
    cvp_model = torch.compile(CVPR3(base_model, 3, ssl_model).to(device))
    optimiser = torch.optim.SGD(cvp_model.head.parameters(), lr=0.001)
    return CVPTrainer(cvp_model, optimiser, training_data=train_data, test_data=test_loader, tau=0.2, transforms=_cvp_transforms(), n_views=3)
