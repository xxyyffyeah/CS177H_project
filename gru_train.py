import torch
from convgru import ConvGRU
from gru_model import GruRNN
import torch.nn as nn
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
gru_model = GruRNN(input_size=8, hidden_size=32,
                  output_size=1, num_layers=3)
gru_model.to(device)
# INPUT_FEATURES_NUM = 5
# OUTPUT_FEATURES_NUM = 1
# train_x_tensor = train_x.reshape(-1, 1, INPUT_FEATURES_NUM)  # set batch size to 1
# train_y_tensor = train_y.reshape(-1, 1, OUTPUT_FEATURES_NUM)  # set batch size to 1

# # transfer data to pytorch tensor
# train_x_tensor = torch.from_numpy(train_x_tensor)
# train_y_tensor = torch.from_numpy(train_y_tensor)

#gru_model = GruRNN(INPUT_FEATURES_NUM, 30, output_size=OUTPUT_FEATURES_NUM, num_layers=1)  # 30 hidden units
print('GRU model:', gru_model)
print('model.parameters:', gru_model.parameters)
#print('train x tensor dimension:', Variable(train_x_tensor).size())

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(gru_model.parameters(), lr=1e-2)
"""

prev_loss = 1000
max_epochs = 2000

train_x_tensor = train_x_tensor.to(device)

for epoch in range(max_epochs):
    output = gru_model(train_x_tensor).to(device)
    loss = criterion(output, train_y_tensor)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if loss < prev_loss:
        torch.save(gru_model.state_dict(), 'lstm_model.pt')  # save model parameters to files
        prev_loss = loss

    if loss.item() < 1e-4:
        print('Epoch [{}/{}], Loss: {:.5f}'.format(epoch + 1, max_epochs, loss.item()))
        print("The loss value is reached")
        break
    elif (epoch + 1) % 100 == 0:
        print('Epoch: [{}/{}], Loss:{:.5f}'.format(epoch + 1, max_epochs, loss.item()))

# prediction on training dataset
pred_y_for_train = gru_model(train_x_tensor).to(device)
pred_y_for_train = pred_y_for_train.view(-1, OUTPUT_FEATURES_NUM).data.numpy()
"""
