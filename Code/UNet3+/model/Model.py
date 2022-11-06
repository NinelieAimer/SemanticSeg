from .ModelParts import *
import torch
import torch.nn.functional as F
from utils.init_weights import *


class UNet3Plus(nn.Module):
    def __init__(self,in_channels:int,cls_num:int,) -> None:
        super().__init__()

        filters=[64,128,256,512,1024]
        self.in_channels=in_channels
        self.cls_num=cls_num
        self.CatChannels = filters[0]
        self.CatBlocks = 5
        self.UpChannels=self.CatBlocks*self.CatChannels

        
        #----------Encoder----------
        self.down_conv=nn.ModuleList([DownSample(i,o) for i,o in 
                                      [(in_channels,64),(64,128),(128,256),(256,512),(512,1024)]])
        
        #----------Decoder----------
        
        #D4
        self.d4_e1=ProcessEn(filters[0],self.CatChannels,8)
        self.d4_e2=ProcessEn(filters[1],self.CatChannels,4)
        self.d4_e3=ProcessEn(filters[2],self.CatChannels,2)
        self.d4_e4=ProcessEn(filters[3],self.CatChannels,1)
        
        self.d4_d5=UpDe(filters[4],self.CatChannels,2,'bilinear')        
        
        self.d4=ProcessDe(self.UpChannels,self.UpChannels)
        
        #D3
        self.d3_e1=ProcessEn(filters[0],self.CatChannels,4)
        self.d3_e2=ProcessEn(filters[1],self.CatChannels,2)
        self.d3_e3=ProcessEn(filters[2],self.CatChannels,1)
        
        self.d3_d5=UpDe(filters[4],self.CatChannels,4,mode='bilinear')
        self.d3_d4=UpDe(self.UpChannels,self.CatChannels,2,mode='bilinear')
        
        self.d3=ProcessDe(self.UpChannels,self.UpChannels)

        #D2
        self.d2_e1=ProcessEn(filters[0],self.CatChannels,2)
        self.d2_e2=ProcessEn(filters[1],self.CatChannels,1)
        
        self.d2_d5=UpDe(filters[4],self.CatChannels,8,mode='bilinear')
        self.d2_d4=UpDe(self.UpChannels,self.CatChannels,4,mode='bilinear')
        self.d2_d3=UpDe(self.UpChannels,self.CatChannels,2,mode='bilinear')
        
        self.d2=ProcessDe(self.UpChannels,self.UpChannels)
        
        #D1
        self.d1_e1=ProcessEn(filters[0],self.CatChannels,1)
        
        self.d1_d5=UpDe(filters[4],self.CatChannels,16,mode='bilinear')
        self.d1_d4=UpDe(self.UpChannels,self.CatChannels,8,mode='bilinear')
        self.d1_d3=UpDe(self.UpChannels,self.CatChannels,4,mode='bilinear')
        self.d1_d2=UpDe(self.UpChannels,self.CatChannels,2,mode='bilinear')
        
        self.d1=ProcessDe(self.UpChannels,self.UpChannels)
    
        
        #UpSample Decoder to use deepsup
        self.up5=nn.Upsample(scale_factor=32,mode='bilinear')
        self.up4=nn.Upsample(scale_factor=16,mode='bilinear')
        self.up3=nn.Upsample(scale_factor=8,mode='bilinear')
        self.up2=nn.Upsample(scale_factor=4,mode='bilinear')
        self.up1=nn.Upsample(scale_factor=2,mode='bilinear')
        
        #DeepSup
        self.out1=nn.Conv2d(self.UpChannels,self.cls_num,3,padding=1)
        self.out2=nn.Conv2d(self.UpChannels,self.cls_num,3,padding=1)
        self.out3=nn.Conv2d(self.UpChannels,self.cls_num,3,padding=1)
        self.out4=nn.Conv2d(self.UpChannels,self.cls_num,3,padding=1)
        self.out5=nn.Conv2d(filters[4],self.cls_num,3,padding=1)
        
        #CGM
        self.cls=nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Conv2d(filters[4],2,1),
            nn.AdaptiveMaxPool2d(1),
            nn.Sigmoid()
        )
        
        # #initialise weights
        # for m in self.modules():
        #     if isinstance(m,nn.Conv2d):
        #         m.apply(init_weights)
        #     elif isinstance(m,nn.BatchNorm2d):
        #         m.apply(init_weights)

        

    def forward(self,inputs):
        #downsample
        e1=self.down_conv[0](inputs)
        e2=self.down_conv[1](e1)
        e3=self.down_conv[2](e2)
        e4=self.down_conv[3](e3)
        e5_d5=self.down_conv[4](e4)
        
        #classification
        cls_branch=self.cls(e5_d5)
        cls_branch_max=cls_branch.argmax(dim=1,keepdim=True)
        
        #decoder
        d4_e1=self.d4_e1(e1)
        d4_e2=self.d4_e2(e2)
        d4_e3=self.d4_e3(e3)
        d4_e4=self.d4_e4(e4)
        d4_d5=self.d4_d5(e5_d5)
        d4=self.d4(torch.concat((d4_e1,d4_e2,d4_e3,d4_e4,d4_d5),dim=1))
        
        d3_e1=self.d3_e1(e1)
        d3_e2=self.d3_e2(e2)
        d3_e3=self.d3_e3(e3)
        d3_d4=self.d3_d4(d4)
        d3_d5=self.d3_d5(e5_d5)
        d3=self.d3(torch.concat((d3_e1,d3_e2,d3_e3,d3_d4,d3_d5),dim=1))
        
        d2_e1=self.d2_e1(e1)
        d2_e2=self.d2_e2(e2)
        d2_d3=self.d2_d3(d3)
        d2_d4=self.d2_d4(d4)
        d2_d5=self.d2_d5(e5_d5)
        d2=self.d2(torch.cat((d2_e1,d2_e2,d2_d3,d2_d4,d2_d5),dim=1))
        
        d1_e1=self.d1_e1(e1)
        d1_d2=self.d1_d2(d2)
        d1_d3=self.d1_d3(d3)
        d1_d4=self.d1_d4(d4)
        d1_d5=self.d1_d5(e5_d5)
        d1=self.d1(torch.concat((d1_e1,d1_d2,d1_d3,d1_d4,d1_d5),dim=1))

        #Upsampling
        up_d5=self.up5(e5_d5)
        up_d4=self.up4(d4)
        up_d3=self.up3(d3)
        up_d2=self.up4(d2)
        up_d1=self.up1(d1)
        
        #deepsup
        out1=self.out1(up_d1)
        out2=self.out2(up_d2)
        out3=self.out3(up_d3)
        out4=self.out4(up_d4)
        out5=self.out5(up_d5)
        
        #add CGM
        out1=out1*cls_branch_max
        out2=out2*cls_branch_max
        out3=out3*cls_branch_max
        out4=out4*cls_branch_max
        out5=out5*cls_branch_max
        
        return F.sigmoid(out1).float(),F.sigmoid(out2).float,F.sigmoid(out3),F.sigmoid(out4),F.sigmoid(out5)
        
        
        
    

        