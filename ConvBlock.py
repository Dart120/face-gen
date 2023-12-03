import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from EqConv2d import EqConv2D
from PixelNorm import PixelNorm
from Sampling import upsample


class ConvBlock(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(ConvBlock, self).__init__()
        self.conv_1 = EqConv2D(in_channels,out_channels,(3,3))
        self.conv_2 = EqConv2D(out_channels,out_channels,(3,3))
        self.pn = PixelNorm()
    def forward(self,x):
        # different
        x = upsample(x) 
        x = self.conv_1(x)
        x = F.leaky_relu(x,0.2)
        x = self.pn(x)
        x = self.conv_2(x)
        x = F.leaky_relu(x,0.2)
        x = self.pn(x)
        return x
if __name__ == '__main__':
    t = torch.ones((64,512,4,4))
    layer = ConvBlock(512,4)
    print(layer(t).shape)