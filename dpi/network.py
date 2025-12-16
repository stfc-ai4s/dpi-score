"""
    Authors:
        Niraj Bhujel, SciML-STFC-UKRI (niraj.bhujel@stfc.ac.uk) 
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .cnn import CNN3D

class ClassHead(nn.Module):
    """
    Simple MLP network with dropout.
    """
    def __init__(self, in_channels=512, out_channels=2, act_layer=nn.ReLU, dropout=0.0, bias=True):
        super().__init__()

        
        self.act = act_layer()
        self.drop = nn.Dropout(dropout)
        # self.fc0 = nn.Linear(in_channels, in_channels, bias=bias)
        self.fc1 = nn.Linear(in_channels, in_channels//2, bias=bias)
        self.fc2 = nn.Linear(in_channels//2, in_channels//4, bias=bias)
        self.fc3 = nn.Linear(in_channels//4, out_channels, bias=bias)

    def forward(self, x):

        # x = self.drop(self.act(self.fc0(x)))
        x = self.drop(self.act(self.fc1(x)))
        x = self.drop(self.act(self.fc2(x)))
        x = self.fc3(x)
        
        return x
        
class Network(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        
        self.cfg = cfg
        
        self.net = CNN3D(
            len(cfg.data.grid_features), 
            cfg.net.hidden_dim,
            drop_prob=cfg.net.drop,
        )
        
        # Pooling and Flattening
        if cfg.net.pool=='avg':
            self.pool3d = nn.AdaptiveAvgPool3d(output_size=(1, 1, 1))
            
        elif cfg.net.pool=='max':
            self.pool3d = nn.AdaptiveMaxPool3d(output_size=(1, 1, 1))
        else:
            raise Exception(f"cfg.net.pool can be either 'avg' or 'max', but{cfg.net.pool} given!")
            
        self.flat = nn.Flatten(start_dim=1)
        
        # Classification Head
        self.class_head = ClassHead(
            cfg.net.hidden_dim, 
            cfg.data.num_class,
            dropout=cfg.net.drop_head,
        )
    
    def forward(self, batch, **kwargs):

        x = self.net(batch['grids'], **kwargs)
        x = self.pool3d(x)
        
        x = self.class_head(self.flat(x))

        return x
