import torch
import torch.nn as nn


class CVPHead(nn.Module):
    def __init__(self, kernel_size, init='fixed'):
        super().__init__()
        self.conv = nn.Conv2d(3, 3, kernel_size, padding=kernel_size // 2)
        self.lam = nn.Parameter(torch.tensor(1.0))
        with torch.no_grad():
            if init == 'fixed':
                kernel = torch.tensor([[0., -1., 0.], [-1., 5., -1.], [0., -1., 0.]])
                self.conv.weight.zero_()
                for i in range(3):
                    self.conv.weight[i, i] = kernel
            else:
                nn.init.uniform_(self.conv.weight, -1, 1)
            self.conv.bias.zero_()

    def forward(self, x):
        return (x + self.lam * self.conv(x)).clamp(0, 1)


class CVPF3(nn.Module):
    def __init__(self, backbone, kernel_size, ssl_model):
        super().__init__()
        self.head = CVPHead(kernel_size, init='fixed')
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


class CVPR3(nn.Module):
    def __init__(self, backbone, kernel_size, ssl_model):
        super().__init__()
        self.head = CVPHead(kernel_size, init='random')
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
