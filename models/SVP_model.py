# FIX: SVPHead architecture should be Linear(512,128) -> BN(128) -> ReLU -> Linear(128,16)
# FIX: floor/y_s should be built from view index not class label: labels = torch.cat([torch.arange(bs) for _ in range(n_views)], dim=0); y_s = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
# FIX: den should exp first then sum: den = torch.exp(sim / tau).sum(dim=1), current order (sum then exp) is wrong
# FIX: tau should be 0.2

import torch
import torch.nn as nn
import torch.nn.functional as F


class SVPHead(nn.Module):
    # TODO: Two layer MLP, idk what size exactly
    def __init__(self):
        super().__init__()
        # NOTE: Double check the resnet size
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
        )

    def forward(self, x):

        return self.linear_relu_stack(x)


class SVP(nn.Module):
    # So we take the RESNET, freeze it. Give it an two layer MLP head.
    def __init__(self, backbone):
        super().__init__()
        self.head = SVPHead()
        self.backbone = backbone
        for param in self.backbone.parameters():
            param.requires_grad = False

    def forward(self, x):
        x_hat = self.backbone(x)
        return self.head(x_hat)
