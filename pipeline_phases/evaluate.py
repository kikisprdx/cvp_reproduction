import argparse
import timm
import torch
from pipeline_phases.training_SSL_pipeline import testing_phase_standard, training_phase_SSL, testing_phase_prompting
from data.data_loader import get_data_loader_CIFAR10C, get_data_loader_CIFAR10
from models.SSL_model import SSL_model

def load_resnet26_model():
    model = timm.create_model('resnet26', pretrained=False, num_classes=10)
    best_model_path = "models/best_resnet26.pth"
    model.load_state_dict(torch.load(best_model_path, map_location=torch.device('cpu')))
    return model


def main():
    train_loader = get_data_loader_CIFAR10(batch_size=64)  # Clean data for SSL training
    test_loader = get_data_loader_CIFAR10C(batch_size=16)  # Corrupted data (Paper uses batch size 16 for testing it seems)

    resnet26 = load_resnet26_model()
    
    ssl_model = SSL_model(in_dim=2048, hidden=256, out_dim=128) 

    # train the SSL Model (Offline Phase): phase 1
    print("----- SSL Training -----")
    training_phase_SSL(resnet26, ssl_model, train_loader, epochs=200)

    # baseline : phase 2
    print("----- Standard Baseline Evaluation -----")
    standard_acc = testing_phase_standard(resnet26, test_loader)
    print(f"Standard Accuracy on Corrupted Data: {standard_acc:.4f}")

    '''
    # test-time adaptation phase: phase 3
    print("--- Starting Phase 3: CVP Prompting Evaluation ---")
    ssl_model.load_state_dict(torch.load('models/ssl_weights.pth'))
    cvp_acc = testing_phase_prompting(resnet26, ssl_model, test_loader)
    print(f"CVP Accuracy on Corrupted Data: {cvp_acc:.4f}")

    print(f"Total Accuracy Improvement: {(cvp_acc - standard_acc) * 100:.2f}%")
    '''

if __name__ == '__main__':
    main()