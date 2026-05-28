# FIX: scripted_transforms for multi-view augmentation needs to be defined here and passed to trainer

import timm
import torch
from torchvision import transforms

from data.data_loader import get_data_loader_CIFAR10, get_data_loader_CIFAR10C_generated
from models.SVP_model import SVP
from pipeline_phases.training_SVP_pipeline import SVPTrainer


def main():
    batch_size = 512
    best_model_path = "models/best_resnet26.pth"
    model = timm.create_model("resnet26", pretrained=False, num_classes=10)
    model.load_state_dict(torch.load(best_model_path))
    model.reset_classifier(0)
    model.eval()
    svp_model = SVP(model)

    # Augmentation Transforms:
    size = 32
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
    optimiser = torch.optim.Adam(svp_model.parameters(), lr=1e-4)
    train_data = get_data_loader_CIFAR10(batch_size=batch_size)
    # print(train_data.dataset[0])
    test_data = get_data_loader_CIFAR10C_generated(
        "datasets/cifar10c_gen.parquet", batch_size=batch_size
    )
    trainer = SVPTrainer(svp_model, optimiser, train_data, test_data, transforms=data_transforms)

    for t in range(200):
        print(f"Epoch {t + 1}\n", "-" * 10)
        trainer.train(batch_size)


if __name__ == "__main__":
    main()
