import torch
from torch import nn
from gru_model import GruRNN
import numpy as np

INPUT_FEATURES_NUM =  24
OUTPUT_FEATURES_NUM = 1

VIRUS_SIZE = 500
HOST_SIZE = 500

PATH = 'gru_model_1.98loss.pt'
model =  GruRNN(input_size=INPUT_FEATURES_NUM, hidden_size=32,output_size=OUTPUT_FEATURES_NUM, num_layers=3)
model.load_state_dict(torch.load(PATH))
model.eval()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
x = np.load('virus.npy').astype('float32')
x = torch.from_numpy(x)

output = model(x[1].unsqueeze(0)).to(device)
data_v_y = torch.zeros(10, dtype = torch.float)
data_h_y = torch.ones(10, dtype = torch.float)
data_y = torch.concatenate((data_v_y, data_h_y), axis = 0)
y = data_y[0].unsqueeze(0).unsqueeze(0)
print(output.shape)  

print(y.shape)