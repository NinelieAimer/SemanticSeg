import torch.nn as nn
from .ModelParts import *

class MobileNetV3_small(nn.Module):
    def __init__(self,in_channels,out_classes) -> None:
        super().__init__()
        
        self.down=nn.Sequential(
            nn.Conv2d(in_channels,16,kernel_size=3,stride=2,padding=1,bias=False),
            nn.BatchNorm2d(16),
            hsigmoid(),
        )
        
        self.bneck=nn.Sequential(
            Block(16,16,16,3,nn.ReLU(inplace=True),2,semodule=SEmodule(16)),
            Block(16,72,24,3,nn.ReLU(inplace=True),2),
            Block(24,88,24,3,nn.ReLU(inplace=True),1),
            Block(24,96,40,5,hsigmoid(),2,semodule=SEmodule(96)),
            Block(40,240,40,5,hsigmoid(),1,SEmodule(240)),
            Block(40,240,40,5,hsigmoid(),1,semodule=SEmodule(240)),
            Block(40,120,48,5,hsigmoid(),1,SEmodule(120)),
            Block(48,144,48,5,hsigmoid(),1,SEmodule(144)),
            Block(48,288,96,5,hsigmoid(),2,SEmodule(288)),
            Block(96,576,96,5,hsigmoid(),1,SEmodule(576)),
            Block(96,576,96,5,hsigmoid(),1,SEmodule(576)),
        )
        
        self.expand=nn.Sequential(
            nn.Conv2d(96,576,1,1,bias=False),
            nn.BatchNorm2d(576),
            hsigmoid()           
        )
        

        self.expand2=nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(576,1024,1,bias=False),
            hsigmoid()
        )        
        
        self.cls=nn.Conv2d(1024,out_classes,bias=False)
        
    def forward(self,inputs):
        out=self.down(inputs)
        out=self.bneck(out)
        out=self.expand(out)
        out=self.expand2(out)
        out=self.cls(out)
        
        return out