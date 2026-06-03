import argparse

import timm
import torch

from data.data_loader import get_data_loader_CIFAR10, get_data_loader_CIFAR10C
from models.SSL_model import SSL_model
from models.SVP_model import SVP
from pipeline_phases.training_phases_manager import (
    testing_phase_prompting,
    testing_phase_standard,
    training_phase_SSL,
)


def load_resnet26_model():
    model = timm.create_model("resnet26", pretrained=False, num_classes=10)
    best_model_path = "models/best_resnet26.pth"
    model.load_state_dict(torch.load(best_model_path, map_location=torch.device("cpu")))
    return model


def setup_svp_eval():
    batch_size = 512
    best_model_path = "models/best_resnet26.pth"
    model = timm.create_model("resnet26", pretrained=False, num_classes=10)
    model.load_state_dict(torch.load(best_model_path))
    svp_model = SVP(model)
    svp_model.load_state_dict(torch.load(best_model_path))

    # Augmentation Transforms:
    optimiser = torch.optim.Adam(svp_model.parameters(), lr=1e-4)
    train_data = get_data_loader_CIFAR10(batch_size=batch_size)
    # print(train_data.dataset[0])
    test_data = get_data_loader_CIFAR10C_generated(
        "datasets/cifar10c_gen.parquet", batch_size=batch_size
    )
    trainer = SVPTrainer(svp_model, optimiser, train_data, test_data)

    for t in range(1):
        print(f"Epoch {t + 1}\n", "-" * 10)
        trainer.test_loop()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["training", "testing"], required=True)
    parser.add_argument(
        "--model", choices=["baseline", "CVP", "SVP", "FT"], default="baseline"
    )
    args = parser.parse_args()

    resnet26 = load_resnet26_model()
    train_loader = get_data_loader_CIFAR10(batch_size=64)  # Clean data for SSL training
    test_loader = get_data_loader_CIFAR10C(
        batch_size=16
    )  # Corrupted data (Paper uses batch size 16 for testing it seems)

    # if arg mode == training
    # train the SSL Model (Offline Phase): phase 1
    if args.mode == "training":
        ssl_model = SSL_model(in_dim=2048, hidden=256, out_dim=128)
        print("----- SSL Training -----")
        training_phase_SSL(resnet26, ssl_model, train_loader, epochs=200)

    elif args.mode == "testing":
        # baseline : phase 2
        # if arg mod == testing and model == baseline
        if args.model == "baseline":
            print("----- Standard Baseline Evaluation -----")
            standard_acc = testing_phase_standard(resnet26, test_loader)
            print(f"Standard Accuracy on Corrupted Data: {standard_acc:.4f}")

        # else do it here:
        else:
            # test-time adaptation phase: phase 3
            print("--- Starting Phase 3: CVP Prompting Evaluation ---")
            ssl_model = SSL_model(in_dim=2048, hidden=256, out_dim=128)
            ssl_model.load_state_dict(torch.load("models/ssl_weights.pth"))
            cvp_acc = testing_phase_prompting(
                resnet26, ssl_model, test_loader, method=args.model
            )
            print(f"CVP Accuracy on Corrupted Data: {cvp_acc:.4f}")


if __name__ == "__main__":
    main()
