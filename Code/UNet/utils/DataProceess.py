import torch
from skimage import io
from torch.utils.data import Dataset


class MyDataSet(Dataset):
    
    #一开始需要什么参数一般就是几个，文件列表，如何处理
    def __init__(self,img_list,label_list) -> None:
        super().__init__()
        self.img_list=img_list
        self.label_list=label_list
    
    #__getitem__方法比较难写，那就放最后，每个写方法，都要记得加self，这里不是C++，不会默认传指针的
    def __len__(self):
        return len(self.img_list)

    #需要复杂处理，单独写静态方法处理比较好，因为我这里处理简单，只是用来示范，不用加标志区分img和label。而且处理过程最好在外部试一下
    @staticmethod
    def preprocess(img):
        img=img/255
        return img
    
    def __getitem__(self, index):
        
        img=io.imread(self.img_list[index])
        label=io.imread(self.label_list[index])
        
        img=self.preprocess(img=img)
        label=self.preprocess(label)
        
        #转为pytorch tensor，记得用copy()
        img_tensor=torch.as_tensor(img.copy()).float().contiguous()
        label_tensor=torch.as_tensor(label.copy()).long().contiguous()
        
        return img_tensor,label_tensor