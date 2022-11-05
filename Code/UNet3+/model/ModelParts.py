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

class DownSample(nn.Module):
    def __init__(self,in_channels:int,out_channels:int,mid_channels=None) -> None:
        super().__init__()
        if not mid_channels:
            mid_channels=out_channels
        
        self.in_channels=in_channels
        self.out_channels=out_channels
        self.mid_channels=mid_channels

        self.conv1=nn.Sequential(
            nn.Conv2d(in_channels,mid_channels,3,1,1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )
        
        self.conv2=nn.Sequential(
            nn.Conv2d(mid_channels,out_channels,3,1,1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        self.max_pooling=nn.MaxPool2d(kernel_size=2)

    def forward(self,x):
        x=self.conv1(x)
        x=self.conv2(x)
        x=self.max_pooling(x)
        return x

class ProcessEn(nn.Module):
    def __init__(self,in_channels:int,cat_channels:int,scale:int) -> None:
        super().__init__()
        
        self.pooling=nn.MaxPool2d(scale,ceil_mode=True)
        self.conv=nn.Conv2d(in_channels,cat_channels,3,1)
        self.bn=nn.BatchNorm2d(cat_channels)
        self.act=nn.ReLU(inplace=True)
    
    def forward(self,x):
        x=self.pooling(x)
        x=self.conv(x)
        x=self.bn(x)
        x=self.act(x)
        return x

class UpDe(nn.Module):
    def __init__(self,in_channels:int,cat_channels:int,scale:int,mode:str) -> None:
        super().__init__()
        
        self.up=nn.Upsample(scale_factor=scale,mode=mode)
        self.conv=nn.Conv2d(in_channels,cat_channels,kernel_size=3,padding=1)
        self.bn=nn.BatchNorm2d(cat_channels)
        self.act=nn.ReLU(inplace=True)
        
    def forward(self,x):
        
        x=self.up(x)
        x=self.conv(x)
        x=self.bn(x)
        x=self.act(x)
        return x