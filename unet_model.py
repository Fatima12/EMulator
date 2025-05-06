""" Obtained from https://github.com/milesial/Pytorch-UNet/tree/master/unet """
""" Full assembly of the parts to form the complete network """

from unet_parts import *


    
    
    
class ConvAE(nn.Module):
    def __init__(self, kernel_size=21, n_mid=4, dropout_p=0.2):
        super(ConvAE, self).__init__()
        
        self.c1 = DoubleConv(1, n_mid, kernel_size=kernel_size, dropout_p=dropout_p)
        self.c2 = DoubleConv(n_mid, n_mid, kernel_size=kernel_size, dropout_p=dropout_p)
        self.c3 = DoubleConv(n_mid, n_mid, kernel_size=kernel_size, dropout_p=dropout_p)
        self.c4 = DoubleConv(n_mid, n_mid, kernel_size=kernel_size, dropout_p=dropout_p)
        self.c5 = DoubleConv(n_mid, n_mid, kernel_size=kernel_size, dropout_p=dropout_p)
        self.c6 = DoubleConv(n_mid, n_mid, kernel_size=kernel_size, dropout_p=dropout_p)
        self.c7 = DoubleConv(n_mid, n_mid, kernel_size=kernel_size, dropout_p=dropout_p)
        self.c8 = DoubleConv(n_mid, n_mid, kernel_size=kernel_size, dropout_p=dropout_p)
        self.c9 = DoubleConv(n_mid, n_mid, kernel_size=kernel_size, dropout_p=dropout_p)
        self.c10 = DoubleConv(n_mid, n_mid, kernel_size=kernel_size, dropout_p=0)
        self.cout = OutConv(n_mid, 1)

    def forward(self, x, y=None):
        x = self.c1(x.unsqueeze(1))
        x = self.c2(x)
        x = self.c3(x)
        x = self.c4(x)
        x = self.c5(x)
        x = self.c6(x)
        x = self.c7(x)
        x = self.c8(x)
        x = self.c9(x)
        x = self.c10(x)
        x = self.cout(x)
        return x.squeeze(1)
    
    
class ConvAE_M(nn.Module):
    def __init__(self, kernel_size=21, n_mid=4, dropout_p=0.2):
        super(ConvAE_M, self).__init__()
        
        self.c1 = DoubleConv(1, n_mid, kernel_size=kernel_size, dropout_p=dropout_p)
        self.c2 = DoubleConv(n_mid, n_mid, kernel_size=kernel_size, dropout_p=dropout_p)
        self.c3 = DoubleConv(n_mid, n_mid, kernel_size=kernel_size, dropout_p=dropout_p)
        self.c4 = DoubleConv(n_mid, n_mid, kernel_size=kernel_size, dropout_p=dropout_p)
        self.c5 = DoubleConv(n_mid, n_mid, kernel_size=kernel_size, dropout_p=dropout_p)
        self.c6 = DoubleConv(n_mid, n_mid, kernel_size=kernel_size, dropout_p=dropout_p)
        self.c7 = DoubleConv(n_mid, n_mid, kernel_size=kernel_size, dropout_p=dropout_p)
        self.c8 = DoubleConv(n_mid, n_mid, kernel_size=kernel_size, dropout_p=dropout_p)
        self.c9 = DoubleConv(n_mid, n_mid, kernel_size=kernel_size, dropout_p=dropout_p)
        self.c10 = DoubleConv(n_mid, n_mid, kernel_size=kernel_size, dropout_p=0)
        self.cout = OutConv(n_mid, 1)

    def forward(self, x, y=None):
        z = torch.cat([x, y], dim=1)
        x = z
        x = self.c1(x.unsqueeze(1))
        x = self.c2(x)
        x = self.c3(x)
        x = self.c4(x)
        x = self.c5(x)
        x = self.c6(x)
        x = self.c7(x)
        x = self.c8(x)
        x = self.c9(x)
        x = self.c10(x)
        x = 0.5 * x[:,:,0:int(x.shape[2]/2),:] + 0.5 * x[:,:,int(x.shape[2]/2):,:]
        x = self.cout(x)
        return x.squeeze(1)

class ConvAE_M_shallow(nn.Module):
    def __init__(self, kernel_size=21, n_mid=4, dropout_p=0.2):
        super(ConvAE_M_shallow, self).__init__()
        
        self.c1 = DoubleConv(1, n_mid, kernel_size=kernel_size, dropout_p=dropout_p)
        self.c2 = DoubleConv(n_mid, n_mid, kernel_size=kernel_size, dropout_p=dropout_p)
        self.c3 = DoubleConv(n_mid, n_mid, kernel_size=kernel_size, dropout_p=dropout_p)
        self.c4 = DoubleConv(n_mid, n_mid, kernel_size=kernel_size, dropout_p=dropout_p)
        self.c5 = DoubleConv(n_mid, n_mid, kernel_size=kernel_size, dropout_p=0)
        self.cout = OutConv(n_mid, 1)

    def forward(self, x, y=None):
        z = torch.cat([x, y], dim=1)
        x = z
        x = self.c1(x.unsqueeze(1))
        x = self.c2(x)
        x = self.c3(x)
        x = self.c4(x)
        x = self.c5(x)
        x = 0.5 * x[:,:,0:int(x.shape[2]/2),:] + 0.5 * x[:,:,int(x.shape[2]/2):,:]
        x = self.cout(x)
        return x.squeeze(1)

    
class ConvAE_M_Loc(nn.Module):
    def __init__(self, kernel_size=21, n_mid=4, dropout_p=0.2):
        super(ConvAE_M_Loc, self).__init__()
        
        self.c1 = DoubleConv(1, n_mid, kernel_size=kernel_size, dropout_p=dropout_p)
        self.c2 = DoubleConv(n_mid, n_mid, kernel_size=kernel_size, dropout_p=dropout_p)
        self.c3 = DoubleConv(n_mid, n_mid, kernel_size=kernel_size, dropout_p=dropout_p)
        self.c4 = DoubleConv(n_mid, n_mid, kernel_size=kernel_size, dropout_p=dropout_p)
        self.c5 = DoubleConv(n_mid, n_mid, kernel_size=kernel_size, dropout_p=dropout_p)
        self.cout = OutConv(n_mid, 1)

    def forward(self, z, y=None):
        x = z + y
        x = self.c1(x.unsqueeze(1))
        x = self.c2(x)
        x = self.c3(x)
        x = self.c4(x)
        x = self.c5(x)
        x = self.cout(x)
        return x.squeeze(1)



class UNet(nn.Module):
    def __init__(self, n_channels=1, n_classes=1, kernel_size=3, bilinear=False, stride=1, dropout_p=0.2):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.init_channels = 1

        
        self.inc = DoubleConv(n_channels, 64,  kernel_size=kernel_size, stride=stride, dropout_p=dropout_p)
        self.down1 = Down(64, 128,             kernel_size=kernel_size, stride=stride, dropout_p=dropout_p)
        self.down2 = Down(128, 256,            kernel_size=kernel_size, stride=stride, dropout_p=dropout_p)
        self.down3 = Down(256, 512,            kernel_size=kernel_size, stride=stride, dropout_p=dropout_p)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor, kernel_size=kernel_size, stride=stride, dropout_p=dropout_p)
        self.up1 = Up(1024, 512 // factor, bilinear=bilinear, dropout_p=dropout_p)
        self.up2 = Up(512, 256 // factor, bilinear=bilinear, dropout_p=dropout_p)
        self.up3 = Up(256, 128 // factor, bilinear=bilinear, dropout_p=dropout_p)
        self.up4 = Up(128, 64, bilinear=bilinear, dropout_p=0)
        self.outc = OutConv(64, n_channels)

    def forward(self, x, y=None):
        
        x1 = self.inc(x.unsqueeze(1))
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        
        return logits.squeeze(1)

    


class UNet_M(nn.Module):
    def __init__(self, n_channels=1, n_classes=1, kernel_size=3, bilinear=False, stride=1, dropout_p=0.2):
        super(UNet_M, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.init_channels = 1

        
        self.inc = DoubleConv(n_channels, 64,  kernel_size=kernel_size, stride=stride, dropout_p=dropout_p)
        self.down1 = Down(64, 128,             kernel_size=kernel_size, stride=stride, dropout_p=dropout_p)
        self.down2 = Down(128, 256,            kernel_size=kernel_size, stride=stride, dropout_p=dropout_p)
        self.down3 = Down(256, 512,            kernel_size=kernel_size, stride=stride, dropout_p=dropout_p)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor, kernel_size=kernel_size, stride=stride, dropout_p=dropout_p)
        self.up1 = Up(1024, 512 // factor, bilinear=bilinear, dropout_p=dropout_p)
        self.up2 = Up(512, 256 // factor, bilinear=bilinear, dropout_p=dropout_p)
        self.up3 = Up(256, 128 // factor, bilinear=bilinear, dropout_p=dropout_p)
        self.up4 = Up(128, 64, bilinear=bilinear, dropout_p=0)
        self.outc = OutConv(64, n_channels)

    def forward(self, x, y=None):
        z = torch.cat([x, y], dim=1)
        x = z
        x1 = self.inc(x.unsqueeze(1))
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = 0.5 * x[:,:,0:int(x.shape[2]/2),:] + 0.5 * x[:,:,int(x.shape[2]/2):,:]
        logits = self.outc(x)

        return logits.squeeze(1)    

    
    
class UNet_M_Loc(nn.Module):
    def __init__(self, n_channels=1, n_classes=1, kernel_size=3, bilinear=False, stride=1, dropout_p=0.2, n_channels_s1=16):
        super(UNet_M_Loc, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.init_channels = 1
        
        self.inc = DoubleConv(n_channels, n_channels_s1,  kernel_size=kernel_size, stride=stride, dropout_p=dropout_p)
        self.down1 = Down(n_channels_s1, n_channels_s1*2,             kernel_size=kernel_size, stride=stride, dropout_p=dropout_p)
        self.down2 = Down(n_channels_s1*2, n_channels_s1*4,            kernel_size=kernel_size, stride=stride, dropout_p=dropout_p)
        self.down3 = Down(n_channels_s1*4, n_channels_s1*8,            kernel_size=kernel_size, stride=stride, dropout_p=dropout_p)
        factor = 2 if bilinear else 1
        self.down4 = Down(n_channels_s1*8, n_channels_s1*16 // factor, kernel_size=kernel_size, stride=stride, dropout_p=dropout_p)
        self.up1 = Up(n_channels_s1*16, n_channels_s1*8 // factor, bilinear=bilinear, dropout_p=dropout_p)
        self.up2 = Up(n_channels_s1*8, n_channels_s1*4 // factor, bilinear=bilinear, dropout_p=dropout_p)
        self.up3 = Up(n_channels_s1*4, n_channels_s1*2 // factor, bilinear=bilinear, dropout_p=dropout_p)
        self.up4 = Up(n_channels_s1*2, n_channels_s1, bilinear=bilinear, dropout_p=0)
        self.outc = OutConv(n_channels_s1, n_channels)

    def forward(self, z, y=None):
        x = z + y
        x1 = self.inc(x.unsqueeze(1))
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)

        return logits.squeeze(1)   


class UNet_M_Loc_strided(nn.Module):
    def __init__(self, n_channels=1, n_classes=1, kernel_size=3, bilinear=False, stride=1, dropout_p=0.2, n_channels_s1=16, height=1, width=1):
        super(UNet_M_Loc_strided, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.init_channels = 1
        
        self.inc = DoubleConv_strided(n_channels, n_channels_s1,  kernel_size=kernel_size, stride=stride, dropout_p=dropout_p, height=height, width=width)
        self.down1 = Down_strided(n_channels_s1, n_channels_s1*2, kernel_size=kernel_size, stride=stride, dropout_p=dropout_p, height=int(height/2), width=int(width/2))
        self.down2 = Down_strided(n_channels_s1*2, n_channels_s1*4, kernel_size=kernel_size, stride=stride, dropout_p=dropout_p, height=int(height/4), width=int(width/4))
        self.down3 = Down_strided(n_channels_s1*4, n_channels_s1*8, kernel_size=kernel_size, stride=stride, dropout_p=dropout_p, height=int(height/8), width=int(width/8))
        factor = 2 if bilinear else 1
        self.down4 = Down_strided(n_channels_s1*8, n_channels_s1*16 // factor, kernel_size=kernel_size, stride=stride, dropout_p=dropout_p, height=int(height/16), width=int(width/16))
        self.up1 = Up_strided(n_channels_s1*16, n_channels_s1*8 // factor, bilinear=bilinear, kernel_size=kernel_size, dropout_p=dropout_p)
        self.up2 = Up_strided(n_channels_s1*8, n_channels_s1*4 // factor, bilinear=bilinear, kernel_size=kernel_size, dropout_p=dropout_p)
        self.up3 = Up_strided(n_channels_s1*4, n_channels_s1*2 // factor, bilinear=bilinear, kernel_size=kernel_size, dropout_p=dropout_p)
        self.up4 = Up_strided(n_channels_s1*2, n_channels_s1, bilinear=bilinear, kernel_size=kernel_size, dropout_p=0)
        self.outc = OutConv(n_channels_s1, n_channels)

    def forward(self, z, y=None):
        x = z + y
        x1 = self.inc(x.unsqueeze(1))
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)

        return logits.squeeze(1)    

class UNet_M_Loc_deepNet(nn.Module):
    def __init__(self, n_channels=1, n_classes=1, kernel_size=3, bilinear=False, stride=1, dropout_p=0.2, n_channels_s1=16):
        super(UNet_M_Loc_deepNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.init_channels = 1
        
        self.inc = DoubleConv(n_channels, n_channels_s1,  kernel_size=kernel_size, stride=stride, dropout_p=dropout_p)
        self.down1 = Down(n_channels_s1, n_channels_s1*2,             kernel_size=kernel_size, stride=stride, dropout_p=dropout_p)
        self.down2 = Down(n_channels_s1*2, n_channels_s1*4,            kernel_size=kernel_size, stride=stride, dropout_p=dropout_p)
        self.down3 = Down(n_channels_s1*4, n_channels_s1*8,            kernel_size=kernel_size, stride=stride, dropout_p=dropout_p)
        self.down4 = Down(n_channels_s1*8, n_channels_s1*16,            kernel_size=kernel_size, stride=stride, dropout_p=dropout_p)
        self.down5 = Down(n_channels_s1*16, n_channels_s1*32,            kernel_size=kernel_size, stride=stride, dropout_p=dropout_p)
        factor = 2 if bilinear else 1
        self.down6 = Down(n_channels_s1*32, n_channels_s1*64 // factor, kernel_size=kernel_size, stride=stride, dropout_p=dropout_p)
        self.up1 = Up(n_channels_s1*64, n_channels_s1*32 // factor, bilinear=bilinear, dropout_p=dropout_p)
        self.up2 = Up(n_channels_s1*32, n_channels_s1*16 // factor, bilinear=bilinear, dropout_p=dropout_p)
        self.up3 = Up(n_channels_s1*16, n_channels_s1*8 // factor, bilinear=bilinear, dropout_p=dropout_p)
        self.up4 = Up(n_channels_s1*8, n_channels_s1*4 // factor, bilinear=bilinear, dropout_p=dropout_p)
        self.up5 = Up(n_channels_s1*4, n_channels_s1*2 // factor, bilinear=bilinear, dropout_p=dropout_p)
        self.up6 = Up(n_channels_s1*2, n_channels_s1, bilinear=bilinear, dropout_p=0)
        self.outc = OutConv(n_channels_s1, n_channels)

    def forward(self, z, y=None):
        x = z + y
        x1 = self.inc(x.unsqueeze(1))
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.down5(x5)
        x7 = self.down6(x6)
        
        x = self.up1(x7, x6)
        x = self.up2(x, x5)
        x = self.up3(x, x4)
        x = self.up4(x, x3)
        x = self.up5(x, x2)
        x = self.up6(x, x1)        
        logits = self.outc(x)

        return logits.squeeze(1)    


class UNet_M_LC(nn.Module):
    def __init__(self, n_channels=1, n_classes=1, kernel_size=3, bilinear=False, stride=1, dropout_p=0.2, n_channels_s1=16):
        super(UNet_M_LC, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.init_channels = 1

        
        self.inc = DoubleConv(n_channels, n_channels_s1,  kernel_size=kernel_size, stride=stride, dropout_p=dropout_p)
        self.down1 = Down(n_channels_s1, n_channels_s1*2,             kernel_size=kernel_size, stride=stride, dropout_p=dropout_p)
        self.down2 = Down(n_channels_s1*2, n_channels_s1*4,            kernel_size=kernel_size, stride=stride, dropout_p=dropout_p)
        self.down3 = Down(n_channels_s1*4, n_channels_s1*8,            kernel_size=kernel_size, stride=stride, dropout_p=dropout_p)
        factor = 2 if bilinear else 1
        self.down4 = Down(n_channels_s1*8, n_channels_s1*16 // factor, kernel_size=kernel_size, stride=stride, dropout_p=dropout_p)
        self.up1 = Up(n_channels_s1*16, n_channels_s1*8 // factor, bilinear=bilinear, dropout_p=dropout_p)
        self.up2 = Up(n_channels_s1*8, n_channels_s1*4 // factor, bilinear=bilinear, dropout_p=dropout_p)
        self.up3 = Up(n_channels_s1*4, n_channels_s1*2 // factor, bilinear=bilinear, dropout_p=dropout_p)
        self.up4 = Up(n_channels_s1*2, n_channels_s1, bilinear=bilinear, dropout_p=0)
        self.outc = OutConv(n_channels_s1, n_channels)

    def forward(self, x, y=None):
        z = torch.cat([x, y], dim=1)
        x = z
        x1 = self.inc(x.unsqueeze(1))
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = 0.5 * x[:,:,0:int(x.shape[2]/2),:] + 0.5 * x[:,:,int(x.shape[2]/2):,:]
        logits = self.outc(x)

        return logits.squeeze(1)    
    



class WNet(nn.Module):
    def __init__(self, n_channels=1, n_classes=1, kernel_size=3, bilinear=False, stride=1, dropout_p=0.2):
        super(WNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.init_channels = 1

        
        self.inc = DoubleConv(n_channels, 64,  kernel_size=kernel_size, stride=stride, dropout_p=dropout_p)
        self.down1 = Down(64, 128,             kernel_size=kernel_size, stride=stride)
        self.down2 = Down(128, 256,            kernel_size=kernel_size, stride=stride)
        self.down3 = Down(256, 512,            kernel_size=kernel_size, stride=stride)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor, kernel_size=kernel_size, stride=stride)
        self.up1 = Up(1024, 512 // factor, bilinear=bilinear)
        self.up2 = Up(512, 256 // factor, bilinear=bilinear)
        self.up3 = Up(256, 128 // factor, bilinear=bilinear)
        self.up4 = Up(128, 64, bilinear=bilinear)
        self.outc = OutConv(64, n_channels)
        
        
        #W part
        self.W1 = DoubleConv(1, 64,  kernel_size=kernel_size, stride=stride, dropout_p=dropout_p)
        self.W2 = Down(64, 128,      kernel_size=kernel_size, stride=stride)
        self.W3 = Down(128, 256,     kernel_size=kernel_size, stride=stride)
        self.W4 = Down(256, 512,     kernel_size=kernel_size, stride=stride)
        self.W5 = Down(512, 1024,    kernel_size=kernel_size, stride=stride)

    def forward(self, x, y=None):
        
        x1 = self.inc(x.unsqueeze(1))
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        if y is not None:
            y = self.W1(y.unsqueeze(1).float())
            y = self.W2(y)
            y = self.W3(y)
            y = self.W4(y)
            y = self.W5(y)
            x5 = x5 + y
        
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        
        return logits.squeeze(1)
    
    
    
    
    
    
class WNet2(nn.Module):
    def __init__(self, n_channels=1, n_classes=1, kernel_size=3, stride=1, dropout_p=0.2):
        super(WNet2, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.init_channels = 1

        
        self.inc = DoubleConv(n_channels, 64, kernel_size=kernel_size, stride=stride, dropout_p=dropout_p)
        self.down1 = Down(64, 128,            kernel_size=kernel_size, stride=stride, dropout_p=dropout_p)
        self.down2 = Down(128, 256,           kernel_size=kernel_size, stride=stride, dropout_p=dropout_p)
        self.down3 = Down(256, 512,           kernel_size=kernel_size, stride=stride, dropout_p=dropout_p)
        self.down4 = Down(512, 1024,          kernel_size=kernel_size, stride=stride, dropout_p=dropout_p)
        self.up1 = Up3(1024, 512) 
        self.up2 = Up(512, 256)
        self.up3 = Up(256, 128)
        self.up4 = Up(128, 64)
        self.outc = OutConv(64, n_channels)
        
        
        #W part
        self.W1 = DoubleConv(1, 64,  kernel_size=kernel_size, stride=stride, dropout_p=dropout_p)
        self.W2 = Down(64, 128,      kernel_size=kernel_size, stride=stride, dropout_p=dropout_p)
        self.W3 = Down(128, 256,     kernel_size=kernel_size, stride=stride, dropout_p=dropout_p)
        self.W4 = Down(256, 512,     kernel_size=kernel_size, stride=stride, dropout_p=dropout_p)
        self.W5 = Down(512, 1024,    kernel_size=kernel_size, stride=stride, dropout_p=dropout_p)

    def forward(self, x, y):
        
        x1 = self.inc(x.unsqueeze(1))
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
#         import pdb; pdb.set_trace()
        
        y = self.W1(y.unsqueeze(1).float())
        y = self.W2(y)
        y = self.W3(y)
        y = self.W4(y)
        y = self.W5(y)
        
        x = self.up1(x5, x4, y)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        
        return logits.squeeze(1)


class LinearAE_M_Loc_3D(nn.Module):
    def __init__(self, kernel_size=5, n_mid=4, dropout_p=0.2):
        super(LinearAE_M_Loc_3D, self).__init__()
        
        self.c1 = DoubleConv_3D(1, n_mid, kernel_size=kernel_size, dropout_p=dropout_p)
        self.cout = OutConv_3D(n_mid, 1)

    def forward(self, z, y=None):
        x = z + y
        x = self.c1(x.unsqueeze(1))
        x = self.cout(x)
        return x.squeeze(1)


class ConvAE_M_Loc_3D(nn.Module):
    def __init__(self, kernel_size=5, n_mid=4, dropout_p=0.2):
        super(ConvAE_M_Loc_3D, self).__init__()
        
        self.c1 = DoubleConv_3D(1, n_mid, kernel_size=kernel_size, dropout_p=dropout_p)
        self.c2 = DoubleConv_3D(n_mid, n_mid, kernel_size=kernel_size, dropout_p=dropout_p)
        self.c3 = DoubleConv_3D(n_mid, n_mid, kernel_size=kernel_size, dropout_p=dropout_p)
        self.c4 = DoubleConv_3D(n_mid, n_mid, kernel_size=kernel_size, dropout_p=dropout_p)
        self.c5 = DoubleConv_3D(n_mid, n_mid, kernel_size=kernel_size, dropout_p=dropout_p)
        self.cout = OutConv_3D(n_mid, 1)

    def forward(self, z, y=None):
        x = z + y
        x = self.c1(x.unsqueeze(1))
        x = self.c2(x)
        x = self.c3(x)
        x = self.c4(x)
        x = self.c5(x)
        x = self.cout(x)
        return x.squeeze(1)

class UNet_M_Loc_3D(nn.Module):
    def __init__(self, n_channels=1, n_classes=1, kernel_size=3, bilinear=False, stride=1, dropout_p=0.2, n_channels_s1=16):
        super(UNet_M_Loc_3D, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.init_channels = 1
        
        self.inc = DoubleConv_3D(n_channels, n_channels_s1,  kernel_size=kernel_size, stride=stride, dropout_p=dropout_p)
        self.down1 = Down_3D(n_channels_s1, n_channels_s1*2,             kernel_size=kernel_size, stride=stride, dropout_p=dropout_p)
        self.down2 = Down_3D(n_channels_s1*2, n_channels_s1*4,            kernel_size=kernel_size, stride=stride, dropout_p=dropout_p)
        self.down3 = Down_3D(n_channels_s1*4, n_channels_s1*8,            kernel_size=kernel_size, stride=stride, dropout_p=dropout_p)
        factor = 2 if bilinear else 1
        self.down4 = Down_3D(n_channels_s1*8, n_channels_s1*16 // factor, kernel_size=kernel_size, stride=stride, dropout_p=dropout_p)
        self.up1 = Up_3D(n_channels_s1*16, n_channels_s1*8 // factor, bilinear=bilinear, dropout_p=dropout_p)
        self.up2 = Up_3D(n_channels_s1*8, n_channels_s1*4 // factor, bilinear=bilinear, dropout_p=dropout_p)
        self.up3 = Up_3D(n_channels_s1*4, n_channels_s1*2 // factor, bilinear=bilinear, dropout_p=dropout_p)
        self.up4 = Up_3D(n_channels_s1*2, n_channels_s1, bilinear=bilinear, dropout_p=0)
        self.outc = OutConv_3D(n_channels_s1, n_channels)

    def forward(self, z, y=None):
        x = z + y
        x1 = self.inc(x.unsqueeze(1))
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)

        return logits.squeeze(1)    
