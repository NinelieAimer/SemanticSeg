import torch
from skimage import io
from torch.utils.data import Dataset

class MyDataSet(Dataset):
    def __init__(self,img_list,label_list) -> None:
        super().__init__()
        self.img_list=img_list
        self.label_list=label_list
        
    def __len__(self):
        return len(self.img_list)
    
    @staticmethod
    def preprocess(img):
        img=img/255
        return img
    
    def __getitem__(self, index):
        img=io.imread(self.img_list[index])
        label=io.imread(self.label_list[index])
        
        img=self.preprocess(img=img)
        label=self.preprocess(label)
        
        img_tensor=torch.as_tensor(img.copy()).float().contiguous()
        label_tensor=torch.as_tensor(label.copy()).long().contiguous()
        return img_tensor,label_tensor
    
    