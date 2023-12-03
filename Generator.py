import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from ConvBlock import ConvBlock
from G_0 import G_0
from RGB import ToRGB
from Sampling import upsample

# No prev to blend to at the first stage
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

class Generator(nn.Module):
    def __init__(self,layer_list):
        super(Generator, self).__init__()
        self.G_0 = G_0(layer_list[0])
        self.rgb_layers = nn.ModuleList([ToRGB(features) for features in layer_list])
        p1 = 1
        self.blocks = nn.ModuleList()
        self.blocks.append(self.G_0)
        while p1 < len(layer_list):
            self.blocks.append(ConvBlock(layer_list[p1 - 1], layer_list[p1]))
            p1 += 1


   
    def forward(self,x,alpha,stage):
        if alpha == 1:
            for i in range(stage + 1):
                x = self.blocks[i](x)
            return self.rgb_layers[stage](x)
        else:
            for i in range(stage):
                x = self.blocks[i](x)
            x_0 = self.rgb_layers[stage - 1](x)
            x_0 = upsample(x_0)
            x_1 = self.blocks[stage](x)
            x_1 = self.rgb_layers[stage](x_1)
            return self.fade_in(x_0,x_1,alpha)
    def fade_in(self,t_0,t_1,alpha):
        return torch.tanh((1 - alpha) * t_0 + (alpha) * t_1)


if __name__ == '__main__':
    t = torch.ones((16,512,1,1)).to(device)
    layer = Generator([512,512,512,512,256,128,64,32,16]).to(device)
    print(layer(t,0.5,6).shape)

    