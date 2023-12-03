import torch
import torch.nn as nn
import torch.nn.functional as F

from EqConv2d import EqConv2D
from Minibatch_Std_Dev import MinibatchStdDev


class D_0(nn.Module):
    def __init__(self,in_features):
        super(D_0, self).__init__()
        self.MSD = MinibatchStdDev()
        self.conv_1 = EqConv2D(in_features + 1,in_features,(3,3),padding = 1)
        self.conv_2 = EqConv2D(in_features,in_features,(4,4),1,0)
        self.fc = EqConv2D(in_features,1,(1,1),1,0)
        
 
    def forward(self,x):
      
        x = self.MSD(x)
     
        x = self.conv_1(x)
        x = F.leaky_relu(x,0.2)

        x = self.conv_2(x)
      
        x = F.leaky_relu(x,0.2)
        x = self.fc(x)
        return x
if __name__ == '__main__':
    layer = D_0()
    data = torch.ones(size=(64,512,4,4))
    print(layer(data).shape)