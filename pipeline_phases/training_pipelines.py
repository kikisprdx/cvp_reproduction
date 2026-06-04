import os

import torch
from torchvision import transforms

from models.CVP_model import CVP
from pipeline_phases.CVP_trainer import CVPTrainer
from pipeline_phases.SVP_trainer import SVPTrainer


def train_svp(size, svp_model, optimiser, train_data, test_loader):
    color_jitter = transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)
    data_transforms = transforms.Compose(
        [
            transforms.RandomResizedCrop(size=size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([color_jitter], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.GaussianBlur(kernel_size=3),
        ]
    )
    trainer = SVPTrainer(
        svp_model, optimiser, train_data, test_loader, transforms=data_transforms
    )

    checkpoint = "models/svp/svp_entire.pth"
    if os.path.exists(checkpoint):
        print(">> Loading existing SVP model, skipping training.")
        trainer.model = torch.load(checkpoint, weights_only=False)
        return trainer

    for t in range(200):
        print(f"Epoch {t + 1}\n", "-" * 10)
        trainer.train(512)
    return trainer


# TODO: Handle variable kernels please
def train_cvp(ssl_model, base_model, train_data, test_loader):
    cvp_model = CVP(base_model, 128, ssl_model)
    optimiser = torch.optim.Adam(cvp_model.head.parameters(), lr=1e-4)
    trainer = CVPTrainer(cvp_model, optimiser, train_data, test_loader, 0.2)
    return trainer
