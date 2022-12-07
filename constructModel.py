

# 2：构造模型
class RNNClassifier(nn.Module):
    """
    这里的bidirectional就是GRU是不是双向的，双向的意思就是既考虑过去的影响，也考虑未来的影响（如一个句子）
    具体而言：正向hf_n=w[hf_{n-1}, x_n]^T,反向hb_0,最后的h_n=[hb_0, hf_n],方括号里的逗号表示concat。
    """
    def __init__(self, input_size, hidden_size, output_size, n_layers=1, bidirectional=True):
        super(RNNClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.n_directions = 2 if bidirectional else 1 # 双向2、单向1
        
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers,  # 输入维度、输出维度、层数、bidirectional用来说明是单向还是双向
                         bidirectional=bidirectional)
        self.fc = nn.Linear(hidden_size * self.n_directions, output_size)
        
    def __init__hidden(self, batch_size):  # 工具函数，作用是创建初始的隐藏层h0
        hidden = torch.zeros(self.n_layers * self.n_directions,
                            batch_size, self.hidden_size)
        return create_tensor(hidden) # 加载GPU
    
    def forward(self, input, seq_lengths):
        # input shape:B * S -> S * B
        input = input.t()
        batch_size = input.size(1)
        
        hidden = self.__init__hidden(batch_size) # 隐藏层h0
        embedding = self.embedding(input)
        
        # pack them up
        gru_input = pack_padded_sequence(embedding, seq_lengths)  # 填充了可能有很多的0，所以为了提速，将每个序列以及序列的长度给出
        
        output, hidden = self.gru(gru_input, hidden) # 只需要hidden
        if self.n_directions == 2: #双向的，则需要拼接起来
            hidden_cat = torch.cat([hidden[-1], hidden[-2]], dim=1)
        else:
            hidden_cat = hidden[-1] # 单向的，则不用处理
        fc_output = self.fc(hidden_cat) # 最后来个全连接层,确保层想要的维度（类别数）
        return fc_output
