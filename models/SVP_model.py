import torch
import torch.nn as nn


class SVPPatch(nn.Module):
    def __init__(self, backbone, ssl_model, patch_size=16, image_size=32):
        super().__init__()
        self.patch_size = patch_size
        self.prompt = nn.Parameter(torch.zeros(1, 3, patch_size, patch_size))
        self.backbone = backbone
        self.ssl_model = ssl_model
        for param in self.backbone.parameters():
            param.requires_grad = False
        for param in self.ssl_model.parameters():
            param.requires_grad = False

    def apply_prompt(self, x):
        pad = torch.zeros_like(x)
        pad[:, :, :self.patch_size, :self.patch_size] = self.prompt
        return x + pad

    def forward(self, x):
        prompted = self.apply_prompt(x)
        features = self.backbone.forward_features(prompted).mean(dim=(2, 3))
        return self.ssl_model(features)


class SVPPad(nn.Module):
    def __init__(self, backbone, ssl_model, pad_width=4, image_size=32):
        super().__init__()
        self.prompt = nn.Parameter(torch.zeros(1, 3, image_size, image_size))
        mask = torch.ones(1, 1, image_size, image_size)
        mask[:, :, pad_width:image_size - pad_width, pad_width:image_size - pad_width] = 0
        self.register_buffer('mask', mask)
        self.backbone = backbone
        self.ssl_model = ssl_model
        for param in self.backbone.parameters():
            param.requires_grad = False
        for param in self.ssl_model.parameters():
            param.requires_grad = False

    def apply_prompt(self, x):
        return x + self.mask * self.prompt

    def forward(self, x):
        prompted = self.apply_prompt(x)
        features = self.backbone.forward_features(prompted).mean(dim=(2, 3))
        return self.ssl_model(features)
