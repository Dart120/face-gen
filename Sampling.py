import torch.nn.functional as F


def upsample(x, scale_factor=2):
    return F.interpolate(x, scale_factor=scale_factor, mode='nearest')
