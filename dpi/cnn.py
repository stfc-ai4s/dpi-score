"""
    Authors:
        Niraj Bhujel, SciML-STFC-UKRI (niraj.bhujel@stfc.ac.uk) 
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

def print_verbose(text, verbose=0):
    if verbose>0:
        print(text)


class SingleConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, conv_layer=nn.Conv3d, activation=nn.ReLU, norm_layer=nn.BatchNorm3d):
        super().__init__()
        self.single_conv = nn.Sequential(
            conv_layer(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            norm_layer(out_channels),
            activation()
        )
    def forward(self, x):
        return self.single_conv(x)
        
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, conv_layer=nn.Conv3d, activation=nn.ReLU, norm_layer=nn.BatchNorm3d):
        super().__init__()

        self.double_conv = nn.Sequential(
            conv_layer(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            norm_layer(out_channels),
            activation(),
            conv_layer(out_channels, out_channels, kernel_size=1, stride=1, padding=0),
            activation(),
            conv_layer(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding),
            norm_layer(out_channels),
            activation(),
        )

    def forward(self, x):
        return self.double_conv(x)
        
class CNN3D(nn.Module):
    DOWNSAMPLE = 4
    FEATURES = [32, 64, 128, 256]
    def __init__(self, input_dim=4, out_dim=256, drop_prob=0):
        super().__init__()
        self.input_dim = input_dim
        self.out_dim = out_dim
        
        self.maxpool = nn.MaxPool3d(kernel_size=2, stride=2, padding=0)
        self.layer1 = SingleConv(input_dim, self.FEATURES[0], kernel_size=3, stride=1, norm_layer=nn.BatchNorm3d)
        self.layer2 = SingleConv(self.FEATURES[0], self.FEATURES[1], kernel_size=3, stride=1, norm_layer=nn.BatchNorm3d)
        self.layer3 = SingleConv(self.FEATURES[1], self.FEATURES[2], kernel_size=3, stride=1, norm_layer=nn.BatchNorm3d)
        self.layer4 = SingleConv(self.FEATURES[-2], out_dim, kernel_size=3, stride=1, norm_layer=nn.BatchNorm3d)
        self.layer4_skip = DoubleConv(self.FEATURES[2], out_dim, kernel_size=3, stride=1, norm_layer=nn.Identity)
        self.dropout = nn.Dropout3d(p=drop_prob)

    def forward(self, x, verbose=0):

        x = self.layer1(x)
        x = self.maxpool(x)
        x = self.dropout(x)
        print_verbose(f"layer1:{x.shape}", verbose)
            
        x = self.layer2(x)
        x = self.maxpool(x)
        x = self.dropout(x)
        print_verbose(f"layer2: {x.shape}", verbose)
        
        x = self.layer3(x)
        # x = self.maxpool(x)
        x = self.dropout(x)
        print_verbose(f"layer3: {x.shape}", verbose)
            
        x = self.layer4(x) + self.layer4_skip(x)
        x = self.dropout(x)
        print_verbose(f"layer4: {x.shape}", verbose)
        
        return x