import torch
import torch.nn as nn



class SSL_model(torch.nn.Module):
    def __init__(self, in_dim, hidden, out_dim):
        self.layer1 = nn.Linear(in_dim, hidden)
        self.activation = nn.ReLU()
        self.layer2 = nn.Linear(hidden, out_dim)

    def forward(self, x):
        x = self.layer1(x)
        x = self.activation(x)
        output = self.layer2(x)
        return(output)
        

