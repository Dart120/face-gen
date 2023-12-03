import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from EqConv2d import EqConv2D
from Sampling import upsample


class ToRGB(nn.Module):
    def __init__(self,in_channels):
        super(ToRGB, self).__init__()
        self.conv_1 = EqConv2D(in_channels,3,(1,1),1,0)
       
    
    def forward(self,x):
        x = self.conv_1(x)
        return x
class FromRGB(nn.Module):
    def __init__(self,out_channels):
        super(FromRGB, self).__init__()
        self.conv_1 = EqConv2D(3,out_channels,(1,1),1,0)
       
    
    def forward(self,x):
        x = self.conv_1(x)
        return x

     
if __name__ == '__main__':
    t = torch.ones((64,3,4,4))
    layer = FromRGB(512)
    print(layer(t).shape)