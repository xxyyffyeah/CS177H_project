import torch
from gru_model import GruRNN
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from dataSet import MyDataset
#import matplotlib.pyplot as plt

INPUT_FEATURES_NUM =  24 
CLASS_NUM = 1

VIRUS_SIZE = 500
HOST_SIZE = 500

BATCH_SIZE = 10
TEST_BATCH_SIZE = 400
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gru_model = GruRNN(input_size=INPUT_FEATURES_NUM, hidden_size=32,output_size=CLASS_NUM, num_layers=2)
gru_model.to(device)

train_data = MyDataset('virus.npy', 'host.npy')
test_data = MyDataset('virus_val.npy', 'host_val.npy')
train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(dataset=test_data, batch_size=TEST_BATCH_SIZE, shuffle=True)

"""
train_data_ratio = 0.8 

data_size = len(data)
t = np.linspace(0, data_size, data_size + 1)
train_data_len = int(train_data_ratio * data_size)

train = index[:train_data_len]
test = index[train_data_len:]

t_for_training = t[:train_data_len]
t_for_testing = t[train_data_len:]
"""


print('GRU model:', gru_model)
print('model.parameters:', gru_model.parameters)
#print('train x tensor dimension:', Variable(train_x_tensor).size())

criterion = nn.BCELoss()
optimizer = torch.optim.Adam(gru_model.parameters(), lr=1e-2)


prev_loss = 1000
max_epochs = 2000

# data = data.to(device)
# data_y = data_y.to(device)
for epoch in range(max_epochs):
    for iteration, (train_x, train_y) in enumerate(train_loader):
        # x =data[i].unsqueeze(0)
        # y = data_y[i].unsqueeze(0).unsqueeze(0)
        
        train_x = train_x.to(device)
        train_y = train_y.to(device)
        #print(train_x.shape)
        output = gru_model(train_x).to(device)
        loss = criterion(output,train_y.to(torch.float))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        
        if iteration % 20 == 0:
            for iteration1, (test_x, test_y) in enumerate(test_loader):
                test_x = test_x.to(device)

                test_output = gru_model(test_x)
                test_y.reshape(TEST_BATCH_SIZE,1)
                predict_y = torch.zeros(TEST_BATCH_SIZE,1)
                accuracy = 0
                for i in range(TEST_BATCH_SIZE):
                    predict_y[i][0] = 1 if test_output[i][0] > 0.5 else 0
                    if predict_y[i][0] == test_y[i][0]: accuracy += 1
                
                accuracy = accuracy / float(test_y.size(0))
                
                print('epoch:{:<2d} | iteration:{:<4d} | loss:{:<6.4f} | accuracy:{:<4.2f}'.format(epoch, iteration, loss, accuracy))

    if loss < prev_loss:
        torch.save(gru_model.state_dict(), 'gru_model.pt')  # save model parameters to files
        prev_loss = loss

    if loss.item() < 1e-4:
        print('Epoch [{}/{}], Loss: {:.5f}'.format(epoch + 1, max_epochs, loss.item()))
        print("The loss value is reached")
        break
    elif (epoch + 1) % 2 == 0:
        print('Epoch: [{}/{}], Loss:{:.5f}'.format(epoch + 1, max_epochs, loss.item()))

"""
# prediction on training dataset
pred_y_for_train = gru_model(train).to(device)
pred_y_for_train = pred_y_for_train.view(-1, OUTPUT_FEATURES_NUM).data.numpy()


gru_model = gru_model.eval()  # switch to testing model

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