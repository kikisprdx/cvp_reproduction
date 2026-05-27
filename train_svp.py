import timm
import torch

from models.SVP_model import SVP
from pipeline_phases.training_SVP_pipeline import SVPTrainer


def main():

    best_model_path = "../models/best_resnet26.pth"
    model = timm.create_model("resnet26", pretrained=False, num_classes=10)
    model.load_state_dict(torch.load(best_model_path))
    svp_model = SVP(model)

    optimiser = torch.optim.SGD(svp_model.parameters(), lr=0.01)
    trainer = SVPTrainer(svp_model, optimiser)

    for t in range(10):
        print(f"Epoch {t + 1}\n", "-" * 10)
        trainer.train()

if __name__ == "__main__":
    main()
