import torch
import torch.nn as nn

#这个点很重要
from .UnetParts import *

class Unet(nn.Module):
    def __init__(self,n_channels,n_classed):
        super().__init__()
        self.n_channels=n_channels
        self.n_classes=n_classed
        
        #downsample
        self.inc=DoubleCovn(n_channels,64)
        self.down1=Down(64,128)
        self.down2=Down(128,256)
        self.down3=Down(256,512)
        self.down4=Down(512,1024)
        
        #upsample and concat
        self.up1=UpAndConcat(1024,512)
        self.up2=UpAndConcat(512,256)
        self.up3=UpAndConcat(256,128)
        self.up4=UpAndConcat(128,64)

        self.outc = FinalConv(64, n_classed)
        
        
        
    def forward(self,x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits