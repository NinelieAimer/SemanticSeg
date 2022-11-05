from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

class Focal_Loss(nn.Module):
    
    
    def __init__(self,alpha:float,gamma:Optional[float]=2.0,
                 reduction:Optional[str]='none') -> None:
        super().__init__()
        self.alpha:float=alpha
        self.gamma:float=gamma
        self.reduction:Optional[str]=reduction
        self.eps:float=1e-6
    
    def forward(self,
                inputs:torch.Tensor,
                target:torch.Tensor) -> torch.Tensor:
                
                input_soft=F.softmax(inputs,dim=1)+self.eps
                
                weight=torch.pow(1.-input_soft,self.gamma)
                focal=-self.alpha*weight*torch.log(input_soft)
                loss_tmp=torch.sum(target*focal,dim=1)
                
                loss=-1
                if self.reduction=='none':
                    loss=loss_tmp
                elif self.reduction=='mean':
                    loss=torch.mean(loss_tmp)
                elif self.reduction=='sum':
                    loss=torch.sum(loss_tmp)
                return loss

def focal_loss(
    inputs:torch.Tensor,
    target:torch.Tensor,
    alpha:float,
    gamma:Optional[float]=2.0,
    reduction:Optional[str]='none'
) -> torch.Tensor:
    
    return Focal_Loss(alpha,gamma,reduction)(inputs,target)
                

                
                