import torch.nn as nn
import torch.nn.functional as F

class Block(nn.Module):
    def __init__(self,in_channels:int,expand_channels:int,out_channels:int,kernel_size:int,
                 act,stride:int,semodule=None) -> None:
        super().__init__()
        
        self.stride=stride
        self.se=semodule
        
        #Expand
        self.expand=nn.Conv2d(in_channels,expand_channels,1,bias=False)
        self.bn1=nn.BatchNorm2d(expand_channels)
        self.act1=act
        
        #depth_wise convolution
        self.depth_wise=nn.Conv2d(expand_channels,expand_channels,kernel_size=kernel_size,
                                  stride=stride,padding=kernel_size//2,bias=False)
        self.bn2=nn.BatchNorm2d(expand_channels)
        self.act2=act
        
        #Compress the dimension
        self.point_wise=nn.Conv2d(expand_channels,out_channels,1,bias=False)
        self.bn3=nn.BatchNorm2d(out_channels)
        
        #default shortcut
        self.shortcut=nn.Sequential()

        #Only if stride==1 the shortcut would exit
        if stride==1 and in_channels!=out_channels:
            self.shortcut=nn.Sequential(
                nn.Conv2d(in_channels,out_channels,1,bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
        
    def forward(self,x):
        out=self.act1(self.bn1(self.expand(x)))
        out=self.act2(self.bn2(self.depth_wise(out)))
        if self.se!=None:
            out=self.se(out)
        out=self.bn3(self.point_wise(out))
        out=out+self.shortcut(x) if self.stride==1 else out
        return out

class SEmodule(nn.Module):
    def __init__(self,in_channels:int,reduction=4) -> None:
        super().__init__()
        self.se=nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels,in_channels//reduction,1,bias=False),
            nn.BatchNorm2d(in_channels//reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels//reduction,in_channels,1,bias=False),
            nn.BatchNorm2d(in_channels),
            hsigmoid()
        )
    
    def forward(self,x):
        return x*self.se(x)


class hsigmoid(nn.Module):
    def forward(self,x):
        out=x*F.relu6(x+3,inplace=True)/6
        return out