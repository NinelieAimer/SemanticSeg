import torch
import torch.nn as nn

class DoubleCovn(nn.Module):
    
    #这里多了一个mid_channel,其实就是第一层卷积到底要多少层，一般都是和输出层一样的，当不是none时候就让他等于输出层通道数
    def __init__(self,in_channels:int,out_channels:int,mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels=out_channels
        
        # 活用Sequential方法，这个东西可以让代码在forward里面看起来更简单，里面写一系列未实例化类的。
        # 而且我们这里用了padding，方便计算
        # 每一个后面都要记得打逗号
        self.double_conv=nn.Sequential(
            nn.Conv2d(in_channels,mid_channels,padding=1,bias=False),
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




        