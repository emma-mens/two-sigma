import numpy as np
import torch
import torch.nn as nn


class FinanceModel(nn.Module):
    def __init__(self, input_dim, output_dim=1):
        super(CollabModelTwo, self).__init__()
        dropout_p = 0.1
        h1 = 1000
        h2 = 500
        self.linear1 = nn.Linear(input_dim, h1)
        self.relu1 = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(h1)
        self.dropout1 = nn.AlphaDropout(p=dropout_p)
        self.linear2 = nn.Linear(h1, h2)
        self.relu2 = nn.ReLU()
        self.bn2 = nn.BatchNorm1d(h2)
        self.dropout2 = nn.AlphaDropout(p=dropout_p)
        self.out = nn.Linear(h2, output_dim)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu1(x)
        x = self.bn1(x)
#         x = self.dropout1(x)
        x = self.linear2(x)
        x = self.relu2(x)
        x = self.bn2(x)
#         x = self.dropout2(x)
        out = self.out(x)
        return out