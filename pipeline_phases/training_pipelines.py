from torchvision import transforms

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

    for t in range(200):
        print(f"Epoch {t + 1}\n", "-" * 10)
        trainer.train(512)


def train_cvp(): 
    pass
