import argparse
import csv
import os
from datetime import datetime
import torch.nn as nn

import timm
import torch

from data.data_loader import get_data_loader_CIFAR10, get_data_loader_CIFAR10C
from models.SSL_model import SSL_model
from pipeline_phases.training_phases_manager import (testing_phase_prompting,
                                                     testing_phase_standard,
                                                     training_phase_SSL)


_RESULTS_PATH = "results/results.csv"
_FIELDS = ["timestamp", "model", "accuracy", "adapt_iters", "tau", "lam", "n_views"]


def log_result(model, accuracy, **kwargs):
    row = {"timestamp": datetime.utcnow().isoformat(), "model": model, "accuracy": round(accuracy, 6)}
    row.update({k: kwargs.get(k) for k in _FIELDS[3:]})
    write_header = not os.path.exists(_RESULTS_PATH)
    with open(_RESULTS_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def load_resnet26_model():
    model = timm.create_model("resnet26", pretrained=False, num_classes=10)

    # replace the 7x7 stride-2 convolution with a 3x3 stride-1 convolution
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)

    # remove the MaxPool layer replacing  with an Identity layer
    model.maxpool = nn.Identity()

    best_model_path = "models/best_resnet26.pth"
    model.load_state_dict(torch.load(best_model_path, map_location=torch.device("cpu")))
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["training", "testing"], required=True)
    parser.add_argument("--model", choices=["baseline", "CVP-F3", "CVP-R3", "SVP-Patch", "SVP-Pad", "FT", "PFT"], default="baseline")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    resnet26 = load_resnet26_model()
    train_loader = get_data_loader_CIFAR10(batch_size=64)  # clean data for SSL training
    test_loader = get_data_loader_CIFAR10C(batch_size=64)

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
            log_result("baseline", standard_acc)

        # else do it here:
        else:
            # test-time adaptation phase: phase 3
            print(f"--- Starting Phase 3: {args.model} Prompting Evaluation ---")
            ssl_model = SSL_model(in_dim=2048, hidden=256, out_dim=128)
            ssl_model.load_state_dict(torch.load('models/ssl_weights.pth', map_location=device))
            method_acc = testing_phase_prompting(resnet26, ssl_model, test_loader, method=args.model)
            print(f"{args.model} Accuracy on Corrupted Data: {method_acc:.4f}")
            hp = {"adapt_iters": 5}
            if args.model in ("CVP-F3", "CVP-R3", "SVP-Patch", "SVP-Pad"):
                hp["tau"] = 0.2
            if args.model in ("CVP-F3", "CVP-R3"):
                hp["lam"] = 0.5
            if args.model in ("SVP-Patch", "SVP-Pad", "FT", "PFT"):
                hp["n_views"] = 3
            log_result(args.model, method_acc, **hp)


if __name__ == "__main__":
    main()
