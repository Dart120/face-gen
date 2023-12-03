import torch
import torch.nn as nn
import torch.nn.functional as F

from EqConv2d import EqConv2D


class DiscBlock(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(DiscBlock, self).__init__()
    # Changes to be more like conv NOT WHAT WAS IN PAPER
        self.conv_1 = EqConv2D(in_channels,out_channels,(3,3))
        self.conv_2 = EqConv2D(out_channels,out_channels,(3,3))
        self.downsample = nn.AvgPool2d(kernel_size=2, stride=2)
    def forward(self,x):
       
        x = self.conv_1(x)
        x = F.leaky_relu(x,0.2)
        x = self.conv_2(x)
        x = F.leaky_relu(x,0.2)
        x = self.downsample(x)
        return x
if __name__ == '__main__':
    t = torch.ones((64,128,128,128))
    layer = DiscBlock(128,256)
    print(layer(t).shape)