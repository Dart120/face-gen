import torch
import torch.nn as nn


class PixelNorm(nn.Module):
    def __init__(self):
        super(PixelNorm, self).__init__()
        pass
    def forward(self,x):
        N,C,H, W = x.shape
        # Square the input
        x_squared = x ** 2
        x_mean = torch.mean(x_squared,axis = 1,keepdim=True)
        x_corrected = x_mean + 1e-8
        x_sqrt = torch.sqrt(x_corrected)
        result = x/x_sqrt
        return result

if __name__ == '__main__':      
    layer = PixelNorm()
    fm = torch.randint(low = 1, high = 10,size = (1000,3,128,64)).to(torch.float)
    print(layer(fm).shape)