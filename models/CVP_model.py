import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import contrastive_loss


class CVPHead(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.conv = nn.Parameter(torch.rand((kernel_size, kernel_size)))

    def forward(self, x):
        return x @ self.conv.T

#NOTE: This uses the SSL_model now; I think
class CVP(nn.Module):
    def __init__(self, backbone, kernel_size, ssl_model):
        super().__init__()
        self.head = CVPHead(kernel_size)
        self.backbone = backbone
        self.ssl_model = ssl_model
        for param in self.backbone.parameters():
            param.requires_grad = False
        for param in self.ssl_model.parameters():
            param.requires_grad = False

    def forward(self, x):
        with torch.no_grad():
            features = self.backbone.forward_features(x).mean(dim=(2, 3))
            x_hat = self.ssl_model(features)
        return self.head(x_hat)
