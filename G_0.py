import torch
import torch.nn as nn
import torch.nn.functional as F

from EqConv2d import EqConv2D
from EqConvTranspose2d import EqConvTranspose2D
from PixelNorm import PixelNorm

# If it works change this back
class G_0(nn.Module):
    def __init__(self,latent_features):
        super(G_0, self).__init__()
        # self.conv_1 = nn.ConvTranspose2d(latent_features,latent_features,4,1,0)
     
        
        self.conv_1 = EqConvTranspose2D(latent_features,latent_features,(4,4),2,0)
        self.conv_2 = EqConv2D(latent_features,latent_features,(3,3))
        self.pn = PixelNorm()
    def forward(self,x):
        x = self.pn(x)
        x = self.conv_1(x)
  
        x = F.leaky_relu(x,0.2)
    
        x = self.conv_2(x)
      
        x = F.leaky_relu(x,0.2)
        x = self.pn(x)
        return x
if __name__ == '__main__':
    t = torch.ones((64,512,1,1))
    layer = G_0(512)
    print(layer(t).shape)