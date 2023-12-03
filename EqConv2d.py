import torch
import torch.nn as nn
import torch.nn.init as init

# Could be wrong
class EqConv2D(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,stride=1,padding='same'):
        super(EqConv2D, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(size=(out_channels, in_channels, *kernel_size)))
        self.bias = nn.Parameter(torch.Tensor(out_channels))
       
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        fan_in = in_channels * kernel_size[0] * kernel_size[1]
        self.scale = 2 * (fan_in ** (-0.5))

        self.reset_parameters()
    def reset_parameters(self):
        nn.init.normal_(self.weight)
        nn.init.zeros_(self.bias)
    def forward(self,x):
        out = nn.functional.conv2d(x,self.weight * self.scale,self.bias,self.stride,self.padding,1,1)
        return out



