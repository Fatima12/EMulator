import torch
import os
from torch import nn
from time import time
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import utils
import torchvision

# Define linear model
class Linear(nn.Module):
    def __init__(self, d, data_shape):
        super(Linear, self).__init__()
        self.model = nn.Sequential(nn.Linear(2,d),
                                  nn.Unflatten(1, data_shape[1:]))
        
    def forward(self, x, y=None):
        return self.model(x)

# Define linear_M model
class Linear_M(nn.Module):
    def __init__(self, d, data_shape):
        super(Linear_M, self).__init__()
        self.model = nn.Sequential(nn.Linear(2+d,d),
                                  nn.Unflatten(1, data_shape[1:]))
        self.d = d
        
    def forward(self, x, y=None):
        y_flat = y.reshape(-1, self.d)
        z = torch.cat([x, y_flat], dim = 1)
        return self.model(z)
    

# define model with loc of antenna on MRI
class Linear_M_Loc(nn.Module):
    def __init__(self, d, data_shape):
        super(Linear_M_Loc, self).__init__()
        self.model = nn.Sequential(nn.Linear(d, d),
                                  nn.Unflatten(1, data_shape[1:]))
        self.d = d
        
    def forward(self, x, y=None):
        z = x + y
        z_flat = z.reshape(-1, self.d)
        return self.model(z_flat)


#Define Linear autoencoder model
class LinearAE(nn.Module):
    def __init__(self, d, n_bottleneck):
        super(LinearAE, self).__init__()
        self.model = nn.Sequential(nn.Flatten(),
                                  nn.Linear(d,n_bottleneck),
                                  nn.Linear(n_bottleneck, d),
                                  )
       
    def forward(self, x, y=None):
        shape = x.shape #Ask lolo
        return self.model(x).view(shape)
    
    

#Define Linear autoencoder_M model
class LinearAE_M(nn.Module):
    def __init__(self, d, n_bottleneck):
        super(LinearAE_M, self).__init__()
        self.model = nn.Sequential(nn.Linear(2*d,n_bottleneck),
                                   nn.Linear(n_bottleneck, d),
                                  )
        self.d = d     
       
    def forward(self, x, y=None): 
        z = torch.cat([nn.Flatten()(x), nn.Flatten()(y)], dim=1)
        shape = x.shape #Ask lolo
        return self.model(z).view(shape)    
    

class LinearAE_M_Loc(nn.Module):
    def __init__(self, d, n_bottleneck):
        super(LinearAE_M_Loc, self).__init__()
        self.model = nn.Sequential(nn.Linear(d,n_bottleneck),
                                   nn.Linear(n_bottleneck, d),
                                  )
        self.d = d     
       
    def forward(self, x, y=None): 
        z = nn.Flatten()(x) + nn.Flatten()(y)
        shape = x.shape #Ask lolo
        return self.model(z).view(shape)       


# define model with loc of antenna on MRI
class Linear_M_Loc_3D(nn.Module):
    def __init__(self, kernel_size = 5, out_channels=4, stride = 1, dilation=1):
        super(Linear_M_Loc_3D, self).__init__()
                #Calculate padding
        padding = int( (dilation*(kernel_size-1) + 1 - stride) / 2 )

        self.model = torch.nn.Conv3d(1, out_channels, kernel_size = kernel_size, padding=padding, stride = stride)
        self.cout = torch.nn.Conv3d(out_channels, 1, kernel_size = 1)
        
    def forward(self, x, y=None):
        z = x + y
        z = z.unsqueeze(1)
        z = self.model(z)
        z = self.cout(z)        
        return z.squeeze(1)



