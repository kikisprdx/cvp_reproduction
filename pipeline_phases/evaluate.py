import argparse
import timm
import torch
from pipeline_phases.training_SSL_pipeline import testing_phase_standard, training_phase_SSL, testing_phase_prompting
from data.data_loader import get_data_loader_CIFAR10C


def load_resnet26_model():
    model = timm.create_model('resnet26', pretrained=False, num_classes=10)
    best_model_path = "models/best_resnet26.pth"
    model.load_state_dict(torch.load(best_model_path, map_location=torch.device('cpu')))
    return model


def main():
    print("hey")
    test_loader = get_data_loader_CIFAR10C(64) # im so confused with your functions

    resnet26 = load_resnet26_model()
    resnet_standard_results = testing_phase_standard(resnet26, test_loader)
    print(resnet_standard_results)

if __name__ == '__main__':
    main()