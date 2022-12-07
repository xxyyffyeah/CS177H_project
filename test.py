import torch
from gru_model import GruRNN
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader

import torchvision
#import matplotlib.pyplot as plt

INPUT_FEATURES_NUM =  24
OUTPUT_FEATURES_NUM = 1
BATCH_SIZE = 1

VIRUS_SIZE = 500
HOST_SIZE = 500
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

train_data = MyDataset('virus.npy', 'host.npy')
train_loader = DataLoader(dataset=train_data, batch_size=10, shuffle=True)
for iteration, (train_x, train_y) in enumerate(train_loader):    # train_x‘s shape (BATCH_SIZE,1,28,28)
    #rain_x = train_x.squeeze()# after squeeze, train_x‘s shape (BATCH_SIZE,28,28),
    #print(train_x.size())  # 第一个28是序列长度(看做句子的长度)，第二个28是序列中每个数据的长度(词纬度)。
  
    print(train_x.shape)
   

