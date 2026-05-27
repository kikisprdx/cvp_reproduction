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
    def loss(self, x, y: torch.Tensor, y_hat: torch.Tensor, tau):
        # y_i_j
        floor = (y.unsqueeze(1) == y.unsqueeze(0)).float()
        # print(floor.size)
        # print(x.size())
        sim = x @ x.mT
        # print("Cosine: ", sim[0])
        # print("Cosine: ", sim.size())
        num = sim / tau
        # print("Tau: ", num[0])
        # print("Tau: ", num.size())
        num = torch.exp(num)
        # print("Log: ", num[0])
        # print("Log: ", num.size())

        den = x @ x.mT
        # print("Cosine: ", sim.size())
        den = torch.exp(den)
        den = torch.sum(den / tau, 1)
        # print("Log: ", den.size())

        # print(floor.size(), num.size(), den.size())
        diff = floor * torch.log(num / den)

        return -(diff).mean()
