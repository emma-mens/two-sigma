import numpy as np
import torch
import torch.nn as nn


class FinanceModel(nn.Module):
    def __init__(self, input_dim, output_dim=1, dropout_p=0.1, binary=False):
        super(FinanceModel, self).__init__()
        
        norm_layer = nn.BatchNorm1d
        
        hidden_units = [1000, 2000, 500, 200]
        sequence = [
            nn.Linear(input_dim, hidden_units[0]),
            nn.LeakyReLU(0.2, True),
            norm_layer(hidden_units[0]),
            nn.AlphaDropout(p=dropout_p)
        ]
        
        for n in range(1, len(hidden_units)):
            sequence += [
                nn.Linear(hidden_units[n-1], hidden_units[n]),
                norm_layer(hidden_units[n]),
                nn.LeakyReLU(0.2, True),
                nn.AlphaDropout(p=dropout_p)
            ]
        sequence += [nn.Linear(hidden_units[-1], output_dim)]
        
        if binary:
            sequence += [nn.Sigmoid()]
        self.model = nn.Sequential(*sequence)

    def forward(self, x):
        return self.model(x)