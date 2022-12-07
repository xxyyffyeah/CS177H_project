import torch
from torch import nn
from torch.autograd import Variable
class GruRNN(nn.Module):
    """
        Parameters：
        - input_size: feature size
        - hidden_size: number of hidden units
        - output_size: number of output
        - num_layers: layers of LSTM to stack
    """
 
    def __init__(self, input_size, hidden_size=1, output_size=1, num_layers=1):
        super().__init__()
        
        self.lstm = nn.GRU(input_size, hidden_size, num_layers)  # utilize the GRU model in torch.nn
        self.linear1 = nn.Linear(hidden_size, 16) # 全连接层
        self.linear2 = nn.Linear(16, output_size) # 全连接层

        
 
    def forward(self, _x):
        x, _ = self.lstm(_x) # _x is input, size (seq_len, batch, input_size)
        #s, b, h = x.shape  # x is output, size (seq_len, batch, hidden_size)
        #x = x.view(s * b, h)
        x = x[:, -1, :]
        x = self.linear1(x)
        x = self.linear2(x)
        x = nn.functional.sigmoid(x)
        #x = x.view(s, b, -1)
        return x



