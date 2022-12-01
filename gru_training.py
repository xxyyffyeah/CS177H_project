import torch
from convgru import ConvGRU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = ConvGRU(input_size=8, hidden_sizes=[32,64,16],
                  kernel_sizes=[3, 5, 3], n_layers=3)
model.to(device)

x = torch.FloatTensor(1,8,64,64).cuda()
x.to(device)

x.requires_grad = True
output = model(x)

# output is a list of sequential hidden representation tensors
print(type(output)) # list

# final output size
#print(output[-1].size()) # torch.Size([1, 16, 64, 64])