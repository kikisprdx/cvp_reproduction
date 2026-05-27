import torch
import torch.nn as nn
import torch.nn.functional as F


class SVPHead(nn.Module):
    # TODO: Two layer MLP, idk what size exactly
    def __init__(self):
        super().__init__()
        # NOTE: Double check the resnet size
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(512, 512),
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

    def loss(self, x, y: torch.Tensor, y_hat: torch.Tensor, tau):
        # y_i_j
        floor = 1 if y_hat.unsqueeze(0).float() == y.unsqueeze(0).float() else 0

        num = [
            [F.cosine_similarity(x_i.unsqueeze(0), x_j.unsqueeze(0)) for x_j in x]
            for x_i in x
        ]
        num = num / tau

        den = [
            [F.cosine_similarity(x_i.unsqueeze(0), x_j.unsqueeze(0)) for x_j in x]
            for x_i in x
        ]
        den = den / tau
        den = torch.sum(den)

        return -(floor * (num - den.unsqueeze(1))).mean()
