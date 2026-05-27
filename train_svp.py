import timm
import torch

from adapters.prompt_update import SVP

def main(): 


    best_model_path = "../models/best_resnet26.pth"
    model = timm.create_model('resnet26', pretrained=False, num_classes=10)
    model.load_state_dict(torch.load(best_model_path))
    svp_model = SVP()
    svp_model.load_backbone(model)
