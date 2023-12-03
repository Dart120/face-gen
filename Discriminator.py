import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from D_0 import D_0
from DiscBlock import DiscBlock
from RGB import FromRGB

# No prev to blend to at the first stage
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

class Discriminator(nn.Module):
    def __init__(self,layer_list):
        super(Discriminator, self).__init__()
        
        self.D_0 = D_0(layer_list[0])
        self.rgb_layers = nn.ModuleList([FromRGB(features) for features in layer_list])
        p1 = 1
        self.blocks = nn.ModuleList()
        self.blocks.append(self.D_0)
        while p1 < len(layer_list):
      
            self.blocks.append(DiscBlock(layer_list[p1], layer_list[p1 - 1]))
            p1 += 1
        self.downsample = nn.AvgPool2d(kernel_size=2, stride=2)


   
    def forward(self,x,alpha,stage):
        if alpha == 1:
            x = self.rgb_layers[stage](x)
            for i in range(stage,-1,-1):
                x = self.blocks[i](x)
            return x.view(x.shape[0], -1)
        else:
            x_1 = self.rgb_layers[stage](x)
            x_1 = self.blocks[stage](x_1)
            x_0 =  self.downsample(x)
            x_0 = self.rgb_layers[stage - 1](x_0)
            x = self.fade_in(x_0,x_1,alpha)
            for i in range(stage - 1,-1,-1):
                x = self.blocks[i](x)
            # Changed
            return x.view(x.shape[0], -1)
        return x
    def fade_in(self,t_0,t_1,alpha):
        return (1 - alpha) * t_0 + (alpha) * t_1


if __name__ == '__main__':
    t = torch.ones((3,3,64,64)).to(device)
    layer = Discriminator([512,512,512,512,256,128,64,32,16]).to(device)
    print(layer(t,0.5,4).shape)