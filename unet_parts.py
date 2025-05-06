""" Obtained from https://github.com/milesial/Pytorch-UNet/tree/master/unet """
""" Parts of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, kernel_size=3, mid_channels=None, leaky=True, stride=1, dilation=1, dropout_p=0.2):
        super().__init__()
        #assert that the kernel is odd
        assert kernel_size & 0x1, 'The kernel must be odd.'
        
        #Calculate padding
        padding = int( (dilation*(kernel_size-1) + 1 - stride) / 2 )
        
        if leaky:
            act = nn.LeakyReLU(inplace=True)
        else:
            act = nn.ReLU(inplace=True)
        
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=kernel_size, padding=padding, bias=False, stride=stride),
            nn.BatchNorm2d(mid_channels),
            act,
            nn.Dropout2d(dropout_p),
            nn.Conv2d(mid_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False, stride=stride),
            nn.BatchNorm2d(out_channels),
            act,
            nn.Dropout2d(dropout_p)            
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dropout_p=0.2):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, kernel_size=kernel_size, stride=stride, dropout_p=dropout_p)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=False, dropout_p=0.2):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, mid_channels=in_channels // 2, dropout_p=dropout_p)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels, dropout_p=dropout_p)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)
    
    
class Up3(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=False, dropout_p=0.2):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, mid_channels=in_channels // 2, dropout_p=dropout_p)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(int(in_channels/2*3), out_channels, dropout_p=dropout_p) # Since we have 3 inputs

    def forward(self, x1, x2, x3):
        x1 = self.up(x1)
        x3 = self.up(x3)

        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x3 = F.pad(x3, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        #Repeat x3 along batch index
        x3 = x3.repeat(x1.shape[0],1,1,1)
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x3, x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
    
####### 3D DoubleConv and OutConv
class DoubleConv_3D(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, kernel_size=3, mid_channels=None, leaky=True, stride=1, dilation=1, dropout_p=0.2):
        super().__init__()
        #assert that the kernel is odd
        assert kernel_size & 0x1, 'The kernel must be odd.'
        
        #Calculate padding
        padding = int( (dilation*(kernel_size-1) + 1 - stride) / 2 )
        
        if leaky:
            act = nn.LeakyReLU(inplace=True)
        else:
            act = nn.ReLU(inplace=True)
        
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size=kernel_size, padding=padding, bias=False, stride=stride),
            nn.BatchNorm3d(mid_channels),
            act,
            nn.Dropout3d(dropout_p),
            nn.Conv3d(mid_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False, stride=stride),
            nn.BatchNorm3d(out_channels),
            act,
            nn.Dropout3d(dropout_p)            
        )

    def forward(self, x):
        return self.double_conv(x)


class OutConv_3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv_3D, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
 
class Down_3D(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dropout_p=0.2):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(2),
            DoubleConv_3D(in_channels, out_channels, kernel_size=kernel_size, stride=stride, dropout_p=dropout_p)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up_3D(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=False, dropout_p=0.2):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv_3D(in_channels, out_channels, mid_channels=in_channels // 2, dropout_p=dropout_p)
        else:
            self.up = nn.ConvTranspose3d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv_3D(in_channels, out_channels, dropout_p=dropout_p)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        
        diffZ = x2.size()[2] - x1.size()[2]        
        diffY = x2.size()[3] - x1.size()[3]
        diffX = x2.size()[4] - x1.size()[4]       
        
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2, 
                        diffZ // 2, diffZ - diffZ // 2])
        
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

#####--------strided versions of the code ----------#####

class DoubleConv_strided(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, kernel_size=3, mid_channels=None, leaky=True, stride=1, dilation=1, dropout_p=0, height=1, width=1):
        super().__init__()
        #assert that the kernel is odd
        assert kernel_size & 0x1, 'The kernel must be odd.'
        
        #Calculate padding
        # padding = int( (dilation*(kernel_size-1) + 1 - stride) / 2 )
        padding_h = math.ceil( (height*stride - stride - height + dilation*(kernel_size-1) + 1) / 2 )
        padding_w = math.ceil( (width*stride - stride - width + dilation*(kernel_size-1) + 1) / 2 )
        
        if leaky:
            act = nn.LeakyReLU(inplace=True)
        else:
            act = nn.ReLU(inplace=True)
        
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=kernel_size, padding=(padding_h, padding_w), bias=False, stride=stride),
            nn.BatchNorm2d(mid_channels),
            act,
            nn.Dropout2d(dropout_p),
            nn.Conv2d(mid_channels, out_channels, kernel_size=kernel_size, padding=(padding_h, padding_w), bias=False, stride=stride),
            nn.BatchNorm2d(out_channels),
            act,
            nn.Dropout2d(dropout_p)            
        )

    def forward(self, x):
        return self.double_conv(x)

class Down_strided(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dropout_p=0.2, height=1, width=1):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv_strided(in_channels, out_channels, kernel_size=kernel_size, stride=stride, dropout_p=dropout_p, height=height, width=width)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up_strided(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=False, kernel_size=3, dropout_p=0.2):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, mid_channels=in_channels // 2, dropout_p=dropout_p)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels, kernel_size=kernel_size, dropout_p=dropout_p)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


