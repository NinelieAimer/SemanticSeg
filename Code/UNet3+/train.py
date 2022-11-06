import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader,random_split
from torch import optim
import os
from utils.FocalLoss import *

from utils.DataProcess import *
from model.Model import *


def train_net(
    net,
    epochs:int=5,
    batch_size:int=1,
    lr:float=1e-5,
    val_percent:float=0.1
):
    dataset=MyDataSet(img_list,label_list)
    
    n_val=int(len(dataset)*val_percent)
    n_train=len(dataset)-n_val
    
    tran_set,val_set=random_split(dataset,[n_train,n_val],generator=torch.Generator().manual_seed(1))
    
    train_loader=DataLoader(dataset,batch_size,True)
    
    optimizer=optim.Adam(net.parameters(),lr=lr)
    
    for epoch in range(1,epochs+1):
        net.train()
        epoch_loss=0
        for i,data in enumerate(train_loader):
            inputs,label=data
            
            inputs=inputs.unsqueeze(0)
             
            mask_pre=net(inputs)
            
            loss=0.
            
            for sub in mask_pre:
                loss+=focal_loss(sub,label,reduction='sum')
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print("epoch:{e}\b \t loss:{l}".format(e=epoch,l=loss))

if __name__=="__main__":
    img_file=os.path.abspath("../../Data/membrane/train/image/")
    label_file=os.path.abspath("../../Data/membrane/train/label")

    img_name=os.listdir(img_file)
    img_list=[img_file+'\\'+str for str in img_name]

    label_name=os.listdir(label_file)
    label_list=[label_file+'\\'+str for str in label_name]

    net=UNet3Plus(1,2)
    train_net(net)
            
            
            