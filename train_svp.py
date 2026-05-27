import timm
import torch

from data.data_loader import get_data_loader_CIFAR10, get_data_loader_CIFAR10C_generated
from models.SVP_model import SVP
from pipeline_phases.training_SVP_pipeline import SVPTrainer


def main():
    batch_size = 512 
    best_model_path = "models/best_resnet26.pth"
    model = timm.create_model("resnet26", pretrained=False, num_classes=10)
    model.load_state_dict(torch.load(best_model_path))
    model.reset_classifier(0)
    svp_model = SVP(model)

    optimiser = torch.optim.SGD(svp_model.parameters(), lr=0.01)
    train_data = get_data_loader_CIFAR10(batch_size=batch_size)
    # print(train_data.dataset[0])
    test_data = get_data_loader_CIFAR10C_generated(
        "datasets/cifar10c_gen.parquet", batch_size=batch_size
    )
    trainer = SVPTrainer(svp_model, optimiser, train_data, test_data)

    for t in range(2):
        print(f"Epoch {t + 1}\n", "-" * 10)
        trainer.train(batch_size)


if __name__ == "__main__":
    main()
