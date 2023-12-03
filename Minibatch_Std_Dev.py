import numpy as np
import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MinibatchStdDev(nn.Module):
    def __init__(self):
        super(MinibatchStdDev, self).__init__()
        pass

    def forward(self,x):
    
        batch_statistics = (
            torch.std(x, dim=0).mean().repeat(x.shape[0], 1, x.shape[2], x.shape[3])
        )
        # we take the std for each example (across all channels, and pixels) then we repeat it
        # for a single channel and concatenate it with the image. In this way the discriminator
        # will get information about the variation in the batch/image
        return torch.cat([x, batch_statistics], dim=1)
        ## (N C + 1 H W)
if __name__ == '__main__':
    layer = MinibatchStdDev()
    fm = torch.randint(low = 1, high = 10,size = (10000,3,128,64)).to(torch.float).to(device)
    print(layer(fm).shape)