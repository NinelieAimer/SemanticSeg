from re import X
from tokenize import Double
from turtle import forward
from typing_extensions import Self
from numpy import diff
from pandas import concat
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

class DoubleCovn(nn.Module):
    
    #这里多了一个mid_channel,如果为None，模型就会把他设置为和输出通道数一样
    def __init__(self,in_channels:int,out_channels:int,mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels=out_channels
        
        # 活用Sequential方法，这个东西可以让代码在forward里面看起来更简单，里面写一系列未实例化类的。
        # 而且我们这里用了padding，方便计算
        # 每一个后面都要记得打逗号
        self.double_conv=nn.Sequential(
            nn.Conv2d(in_channels,mid_channels,kernel_size=3,padding=1,bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True), #这里一定要用inplace=True
            
            nn.Conv2d(mid_channels,out_channels,kernel_size=3,padding=1,bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self,x):
        return self.double_conv(x)



#我们写一下下采样模块，这里其实我们是想把池化和两次卷积结合起来，所以会用到上面的两次卷积
class Down(nn.Module):
    def __init__(self,in_channels,out_channels) -> None:
        super().__init__()
        
        self.maxpool_conv=nn.Sequential(
            nn.MaxPool2d(2),
            DoubleCovn(in_channels,out_channels)
        )
    
    def forward(self,x):
        return self.maxpool_conv(x)





class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self,in_channels,out_channels) -> None:
        super().__init__() 
        
        self.up=nn.ConvTranspose2d(in_channels,out_channels,2,2)
    
    def forward(self,x):
        return self.up(x)

class CropAndConcat(nn.Module):
    def __init__(self) -> None:
        super().__init__()
           
           
    def forward(self,x,contracting_x):
        
        contracting_x = torchvision.transforms.functional.center_crop(contracting_x, [x.shape[2], x.shape[3]])
        return torch.cat([x,contracting_x],dim=1)
    
class UpAndConcat(nn.Module):
    def __init__(self,in_channels,out_channels) -> None:
        super().__init__()
        self.up_conv=Up(in_channels,out_channels)
        self.conat=CropAndConcat()
        self.in_channels=in_channels
        self.out_channels=out_channels
    
    def forward(self,x,contracting_x):
        x=self.up_conv(x)
        x=self.conat(x,contracting_x)
        x=DoubleCovn(x.shape[1],self.out_channels)(x)
        return x

class FinalConv(nn.Module):
    def __init__(self,in_channels,out_channels) -> None:
        super().__init__()
        self.conv=nn.Conv2d(in_channels,out_channels,kernel_size=1)
    
    def forward(self,x):
        return self.conv(x)


        