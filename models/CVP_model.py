import torch.nn as nn


class CVPHead(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.conv = nn.Conv2d(3, 3, kernel_size, padding=kernel_size // 2)

    def forward(self, x):
        return self.conv(x)


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
        prompted = self.head(x)
        features = self.backbone.forward_features(prompted).mean(dim=(2, 3))
        return self.ssl_model(features)
