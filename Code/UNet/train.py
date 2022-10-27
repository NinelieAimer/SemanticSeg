from glob import glob
from sched import scheduler
import torch
import  torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader,random_split
from torch import optim
import os


from utils.DataProceess import MyDataSet
from model import UNetModel

img_file=os.path.abspath("../../Data/membrane/train/image/")
label_file=os.path.abspath("../../Data/membrane/train/label")

img_name=os.listdir(img_file)
img_list=[img_file+'\\'+str for str in img_name]

label_name=os.listdir(label_file)
label_list=[label_file+'\\'+str for str in label_name]

#一般情况下我们都会自定义一个train函数来简化main函数里面代码
def train_net(net,
              epochs:int=5,
              batch_size:int=1,
              learning_rate:float=1e-5,
              val_percent:float=0.1,   #用于交叉验证比例
              save_checkpoint:bool=True
              ):
    
    dataset=MyDataSet(img_list,label_list)
    
    #求出需要多少验证和训练，记得加int，否则乘出来是float类型
    n_val=int(len(dataset)*val_percent)
    n_train=len(dataset)-n_val
    train_set,val_set=random_split(dataset,[n_train,n_val],generator=torch.Generator().manual_seed(0))
    
    #create dataloader，这里num_workers用了多线程技术
    train_loader=DataLoader(train_set,batch_size,True,num_workers=4)
    val_loader=DataLoader(val_set,1,shuffle=False,drop_last=True,num_workers=4)

    #Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    optimizer=optim.Adam(net.parameters(),lr=learning_rate)
    
    #这个是用来动态调整学习率的，很好用
    scheduler=optim.lr_scheduler.ReduceLROnPlateau(optimizer,'max',patience=2)
    
    #默认使用croosentropy来计算损失
    criterion=nn.CrossEntropyLoss()
    global_step=0
    
    #Begin training
    for epoch in range(1,epochs+1):
        print("New Round")
        #将模型设置为训练模式
        net.train()
        epoch_loss=0
        for i,data in enumerate(train_loader):
            inputs,labels=data
            
            #当batch为1的时候需要这一步
            inputs=inputs.unsqueeze(0)
            
            mask_pre=net(inputs)
            
            #这里loss是在循环过程中算的
            mask_pre=F.softmax(mask_pre,dim=1).float()
            label=F.one_hot(labels,2)
            label=label.permute(0,3,1,2).float()
            
            loss=criterion(mask_pre,labels)
            
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
if __name__=="__main__":
    net=UNetModel.Unet(1,2)
    train_net(net)
    
    
    
            
            
            