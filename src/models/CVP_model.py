import torch
import torch.nn as nn
import torch.nn.functional as F


class CVPHead(nn.Module):
    def __init__(self, kernel_size, init='fixed'):
        super().__init__()
        self.kernel = nn.Parameter(torch.zeros(1, 1, kernel_size, kernel_size))
        self.lam = nn.Parameter(torch.tensor(1.0))
        self.padding = kernel_size // 2
        with torch.no_grad():
            if init == 'fixed':
                k = torch.tensor([[0., -1., 0.], [-1., 5., -1.], [0., -1., 0.]])
                self.kernel[0, 0] = k
            else:
                nn.init.uniform_(self.kernel, -1, 1)

    def forward(self, x):
        weight = self.kernel.expand(3, 1, *self.kernel.shape[2:])
        return (x + self.lam * F.conv2d(x, weight, padding=self.padding, groups=3)).clamp(0, 1)


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
