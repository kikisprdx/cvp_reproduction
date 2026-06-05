import torch
import torch.nn as nn


class SVP(nn.Module):
    def __init__(self, backbone, image_size=32):
        super().__init__()
        self.prompt = nn.Parameter(torch.zeros(1, 3, image_size, image_size))
        self.backbone = backbone
        for param in self.backbone.parameters():
            param.requires_grad = False

    def forward(self, x):
        prompted_x = x + self.prompt
        features = self.backbone.forward_features(prompted_x).mean(dim=(2, 3))
        return features
