import torch
import torch.nn as nn


class SSL_model(nn.Module):
    """Two-layer MLP projection head for contrastive SSL feature embedding."""

    def __init__(self, in_dim, hidden, out_dim):
        super().__init__()
        self.layer1 = nn.Linear(in_dim, hidden)
        self.activation = nn.ReLU()
        self.layer2 = nn.Linear(hidden, out_dim)

    def forward(self, x):
        x = self.layer1(x)
        x = self.activation(x)
        output = self.layer2(x)
        return output
        
