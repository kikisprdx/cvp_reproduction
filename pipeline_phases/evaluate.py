from pipeline_phases.training_SVP_pipeline import SVPTrainer
import argparse
import timm
import torch
from models.SVP_model import SVP
from pipeline_phases.training_SSL_pipeline import testing_phase_standard, training_phase_SSL, testing_phase_prompting
from data.data_loader import get_data_loader_CIFAR10, get_data_loader_CIFAR10C, get_data_loader_CIFAR10C_generated


def load_resnet26_model():
    model = timm.create_model('resnet26', pretrained=False, num_classes=10)
    best_model_path = "models/best_resnet26.pth"
    model.load_state_dict(torch.load(best_model_path, map_location=torch.device('cpu')))
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
    print("hey")
    test_loader = get_data_loader_CIFAR10C(64) 

    resnet26 = load_resnet26_model()
    resnet_standard_results = testing_phase_standard(resnet26, test_loader)
    print(resnet_standard_results)

if __name__ == '__main__':
    main()
