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

VIRUS_SIZE = 500
HOST_SIZE = 500
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
gru_model = GruRNN(input_size=INPUT_FEATURES_NUM, hidden_size=32,output_size=OUTPUT_FEATURES_NUM, num_layers=3)
gru_model.to(device)

data = np.concatenate((np.load("virus.npy", mmap_mode='r'),np.load("host.npy", mmap_mode='r')), axis = 0).astype('float32')
data = torch.from_numpy(data)
index = np.random.permutation(len(data))


data_v_y = torch.zeros(VIRUS_SIZE, dtype = torch.float)
data_h_y = torch.ones(HOST_SIZE, dtype = torch.float)
data_y = torch.concatenate((data_v_y, data_h_y), axis = 0)


train_data_ratio = 0.8 

data_size = len(data)
t = np.linspace(0, data_size, data_size + 1)

train_data_len = int(train_data_ratio * data_size)

train = index[:train_data_len]
test = index[train_data_len:]

t_for_training = t[:train_data_len]
t_for_testing = t[train_data_len:]

"""
train_y = np.zeros(train_data_len).astype('float32').reshape(train_data_len,1)

test_size = data_size - train_data_len
test_x = data[train_data_len:]

test_y = np.zeros(data_size - train_data_len).astype('float32').reshape(test_size, 1)

train_x_tensor = train_x.reshape(-1, 298,INPUT_FEATURES_NUM)  # set batch size to 1
#train_x_tensor = train_x
train_y_tensor = train_y.reshape(-1, 1, OUTPUT_FEATURES_NUM)  # set batch size to 1
#print(train_x_tensor)

#train_x_tensor = train_x
#print(train_x_tensor.shape)
#train_y_tensor = train_y
"""


#gru_model = GruRNN(INPUT_FEATURES_NUM, 30, output_size=OUTPUT_FEATURES_NUM, num_layers=1)  # 30 hidden units
print('GRU model:', gru_model)
print('model.parameters:', gru_model.parameters)
#print('train x tensor dimension:', Variable(train_x_tensor).size())

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(gru_model.parameters(), lr=1e-2)


prev_loss = 1000
max_epochs = 2000

data = data.to(device)
data_y = data_y.to(device)
for epoch in range(max_epochs):
    for i in train:
        x =data[i].unsqueeze(0)
        y = data_y[i].unsqueeze(0).unsqueeze(0)
        output = gru_model(x).to(device)
        loss = criterion(output,y.long())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if loss < prev_loss:
        torch.save(gru_model.state_dict(), 'gru_model.pt')  # save model parameters to files
        prev_loss = loss

    if loss.item() < 1.7:
        print('Epoch [{}/{}], Loss: {:.5f}'.format(epoch + 1, max_epochs, loss.item()))
        print("The loss value is reached")
        break
    elif (epoch + 1) % 2 == 0:
        print('Epoch: [{}/{}], Loss:{:.5f}'.format(epoch + 1, max_epochs, loss.item()))


# prediction on training dataset
pred_y_for_train = gru_model(train).to(device)
pred_y_for_train = pred_y_for_train.view(-1, OUTPUT_FEATURES_NUM).data.numpy()


gru_model = gru_model .eval()  # switch to testing model

# prediction on test dataset
test_x_tensor = test_x.reshape(-1, 1,
                                INPUT_FEATURES_NUM)
test_x_tensor = torch.from_numpy(test_x_tensor)  # 变为tensor
test_x_tensor = test_x_tensor.to(device)

pred_y_for_test = gru_model(test_x_tensor).to(device)
pred_y_for_test = pred_y_for_test.view(-1, OUTPUT_FEATURES_NUM).data.numpy()

loss = criterion(torch.from_numpy(pred_y_for_test), torch.from_numpy(test_y))
print("test loss：", loss.item())
"""
# ----------------- plot -------------------
plt.figure()
plt.plot(t_for_training, train_y, 'b', label='y_trn')
plt.plot(t_for_training, pred_y_for_train, 'y--', label='pre_trn')

plt.plot(t_for_testing, test_y, 'k', label='y_tst')
plt.plot(t_for_testing, pred_y_for_test, 'm--', label='pre_tst')

plt.xlabel('t')
plt.ylabel('Vce')

"""