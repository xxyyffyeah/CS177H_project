import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
class MyDataset(Dataset):
    def __init__(self, data1,data2):
        data1 = np.load(data1)
        data2 = np.load(data2)
        self.data = np.concatenate((data1,data2), axis = 0).astype('float32')
        self.lenofdata1 = data1.shape[0]
        self.lenofdata2 = data1.shape[0]
        #self.transforms = torchvision.transforms #转为tensor形式
    def __getitem__(self, index):
        hdct= self.data[index, :, :]  # 读取每一个npy的数据
        hdct = np.squeeze(hdct)  # 删掉一维的数据，就是把通道数这个维度删除
        
        if index < self.lenofdata1:
            ldct = np.ones(1, dtype = int)
        else:
            ldct = np.zeros(1, dtype = int)  
        #ldct = np.squeeze(ldct)
        hdct= torch.from_numpy(hdct)  #转为tensor形式
        ldct= torch.from_numpy(ldct)   #转为tensor形式
        return hdct,ldct #返回数据还有标签
    def __len__(self):
        return self.data.shape[0] #返回数据的总个数