import os
import glob
import torch
import random
import logging
import datetime
import numpy as np
from tqdm import tqdm
import torch.utils.data
import torch.optim as optim
import torch.nn as nn
from common.arguments import opts as parse_args
from common.utils import *
from common.load_data_hm36 import Fusion
from common.h36m_dataset import Human36mDataset
import time
import math

class EBMPrior(nn.Module):
    def __init__(self):
        super().__init__()

        ndf = 1024
        nz = 17*512 + ndf        

        self.featy = nn.Sequential(
            nn.Linear(17*3, ndf),
            nn.Tanh(), #nn.LeakyReLU(0.2),
            nn.Linear(ndf, ndf),
            nn.Tanh(), #nn.LeakyReLU(0.2),
        )

        self.fc1_xy = nn.Linear(nz, ndf)
        self.fc2_xy = nn.Linear(ndf, ndf)
        self.fc3_xy = nn.Linear(ndf, ndf)
        self.fc4_xy = nn.Linear(ndf, 1)
        

    def forward(self, xfeat, y ):
        yfeat = self.featy(y)
        xy_feature = torch.cat((xfeat, yfeat), 1)
        xy_feature = torch.tanh(self.fc1_xy(xy_feature)) # (shape: (batch_size*num_samples, hidden_dim))
        xy_feature = torch.tanh(self.fc2_xy(xy_feature)) + xy_feature # (shape: (batch_size*num_samples, hidden_dim))
        xy_feature = torch.tanh(self.fc3_xy(xy_feature)) + xy_feature # (shape: (batch_size*num_samples, hidden_dim))
        score = self.fc4_xy(xy_feature)

        return score
    
    
def weights_init_xavier(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.xavier_normal_(m.weight)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1, 0.02)
        m.bias.data.fill_(0)


def gauss_density_centered(x, std):
    return torch.exp(-0.5*(x / std)**2) / (math.sqrt(2*math.pi)*std)

def gmm_density_centered(x, std):
    """
    Assumes dim=-1 is the component dimension and dim=-2 is feature dimension. Rest are sample dimension.
    """
    if x.dim() == std.dim() - 1:
        x = x.unsqueeze(-1)
    elif not (x.dim() == std.dim() and x.shape[-1] == 1):
        raise ValueError('Last dimension must be the gmm stds.')
    return gauss_density_centered(x, std).prod(-2).mean(-1)

def sample_gmm_centered(std, num_samples=1):
    num_components = std.shape[-1]
    num_dims = std.numel() // num_components

    std = std.view(1, num_dims, num_components)

    # Sample component ids
    k = torch.randint(num_components, (num_samples,), dtype=torch.int64)
    std_samp = std[0,:,k].t()

    # Sample
    x_centered = std_samp * torch.randn(num_samples, num_dims)
    prob_dens = gmm_density_centered(x_centered, std)

    prob_dens_zero = gmm_density_centered(torch.zeros_like(x_centered), std)

    return x_centered, prob_dens, prob_dens_zero