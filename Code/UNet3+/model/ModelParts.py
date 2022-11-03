from turtle import forward
import torch
import torch.nn as nn
import torch.functional as F

class ProcessDe(nn.Module):
    def __init__(self,in_channels:int,out_channels:int) -> None:
        super().__init__()
        self.in_channels=in_channels
        self.out_channels=out_channels
        self.conv=nn.Conv2d(in_channels,out_channels,3,padding=1)
        self.bn=nn.BatchNorm2d(out_channels)
        self.act=nn.ReLU(inplace=True)
    
    def forward(self,x):
        x=self.conv(x)
        x=self.bn(x)
        x=self.act(x)


class UNet3Plus(nn.Module):
    def __init__(self,in_channels,n_classes) -> None:
        super().__init__()
        
        self.in_channels=in_channels
        self.classes=n_classes
        
        filters=[64,128,256,512,1024]
        
        #Encoder