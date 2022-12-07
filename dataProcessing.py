import torch
import torch.nn as nn
import torch.optim as optim
import csv
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import time
from torch.nn.utils.rnn import pack_padded_sequence # https://pytorch.org/docs/master/generated/torch.nn.utils.rnn.pack_padded_sequence.html
import math

# 1：数据集

# 超参数
HIDDEN_SIZE = 100 # 隐藏层
BATCH_SIZE = 256
N_LAYER = 2 # RNN的层数
N_EPOCHS = 100 # train的轮数
N_CHARS = 128 # 这个就是要构造的字典的长度
USE_GPU = False

class NameDataset(Dataset):  # 这个是自己写的数据集的类，就那3个函数
    def __init__(self, is_train_set=True):
        filename = "E:\\PyTorch\\PyTorch深度学习实践\\names_train.csv" if is_train_set else "E:\\PyTorch\\PyTorch深度学习实践\\names_test.csv"
        with open(filename, "rt") as f:  # 因为这个文件不是很大，所以在初始化的时候就全读进来了
            reader = csv.reader(f)
            rows = list(reader)

    
        self.names = [row[0] for row in rows]
        self.len = len(self.names)
        self.countries = [row[1] for row in rows]
        self.country_list = list(sorted(set(self.countries)))  # 去重+排序
        self.country_dict = self.getCountryDict()  # 做一个国家词典,这个就是标签 y
        self.country_num = len(self.country_list)
        
    def __getitem__(self, index):
        return self.names[index], self.country_dict[self.countries[index]] # 前者是名字字符串，后者是国家的索引
    
    def __len__(self):
        return self.len
    
    def getCountryDict(self):
        country_dict = dict()
        for idx, country_name in enumerate(self.country_list, 0):
            country_dict[country_name] = idx
        return country_dict
    
    def idx2country(self, index): # 这个就是为了得到分类之后，返回下标对应的字符串，也就是显示使用的
        return self.country_list[index]
    
    def getCountriesNum(self):  # 分类的国家数量
        return self.country_num

def make_tensors(names, countries): # 这个就是将名字的字符串转换成数字表示
    sequences_and_lengths = [name2list(name)for name in names]  # [(),(),,...]
    name_sequences = [sl[0] for sl in sequences_and_lengths] # 取转换成ACCIIS的序列,长度是BatchSize
    seq_lengths = torch.LongTensor([sl[1]for sl in sequences_and_lengths]) # 取序列的长度，转换成longtensor
    countries = countries.long() #这个cluntries之前转换成了数字，这里只转换成longtensor

    # make tensor of name, BatchSize x SeqLen
    seq_tensor = torch.zeros(len(name_sequences), seq_lengths.max()).long() #先做全0的张量，然后填充,长度是BatchSize
    for idx, (seq, seq_len) in enumerate(zip(name_sequences, seq_lengths), 0):
        seq_tensor[idx, :seq_len] = torch.LongTensor(seq)

    # sort by length to use pack_padded_sequence
    seq_lengths, perm_idx = seq_lengths.sort(dim=0, descending=True)
    seq_tensor = seq_tensor[perm_idx]
    countries = countries[perm_idx]

    return create_tensor(seq_tensor), \
           create_tensor(seq_lengths), \
           create_tensor(countries)
        
def name2list(name): # 将name字符串的字母转换成ASCII
    arr = [ord(c) for c in name] 
    return arr, len(arr)  # 返回的是元组
    
def create_tensor(tensor):  # 是否使用GPU
    if USE_GPU:
        device = torch.device("cuda:0")
        tensor = tensor.to(device)
    return tensor    
    
trainset = NameDataset(is_train_set=True) # train数据
trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)

testset = NameDataset(is_train_set=False) # test数据
testloader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False)

N_COUNTRY = trainset.getCountriesNum() # 这个就是总的类别的数量
