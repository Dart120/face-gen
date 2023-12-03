import torch
import torch.nn as nn
import torch.nn.init as init


class EqConvTranspose2D(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,stride=1,padding='same'):
        super(EqConvTranspose2D, self).__init__()
        self.weights = nn.Parameter(torch.Tensor(size=(out_channels, in_channels, *kernel_size)))
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
        nn.init.normal_(self.weights)
        nn.init.zeros_(self.bias)
    def forward(self,x):
        out = nn.functional.conv_transpose2d(x,self.weights * self.scale,self.bias,self.stride,self.padding,0,1,1)
        return out


