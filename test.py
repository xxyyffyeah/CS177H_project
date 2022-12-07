import torch
from convgru import ConvGRU
from gru_model import GruRNN
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
import dataset_fuse
#import matplotlib.pyplot as plt

INPUT_FEATURES_NUM =  24
OUTPUT_FEATURES_NUM = 1
BATCH_SIZE = 1
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#gru_model = GruRNN(input_size=INPUT_FEATURES_NUM, hidden_size=32,output_size=OUTPUT_FEATURES_NUM, num_layers=3)
#gru_model.to(device)

data_v_y = torch.zeros(10, dtype = torch.float)
data_h_y = torch.ones(10, dtype = torch.float)
data_y = torch.concatenate((data_v_y, data_h_y), axis = 0)
print(data_y[10].dtype)