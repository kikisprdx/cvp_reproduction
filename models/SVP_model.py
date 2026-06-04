import torch
import torch.nn as nn
import torch.nn.functional as F


class SVPHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 16),
        )

    def forward(self, x):
        return self.linear_relu_stack(x)


class SVP(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.head = SVPHead()
        self.backbone = backbone
        for param in self.backbone.parameters():
            param.requires_grad = False

    def forward(self, x):
        features = self.backbone.forward_features(x).mean(dim=(2, 3))
        return self.head(features)
