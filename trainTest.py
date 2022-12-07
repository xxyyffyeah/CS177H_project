
# 4：训练和测试模型
def trainModel():
    total_loss = 0
    for i, (names, countries) in enumerate(trainloader, 1): # 记载的下标从1开始
        inputs, seq_lengths, target = make_tensors(names, countries)
        output = classifier(inputs, seq_lengths) # 预测输出
        loss = criterion(output, target) # 求出损失
        optimizer.zero_grad() # 清除之前的梯度
        loss.backward() # 梯度反传
        optimizer.step() # 更新参数
        
        total_loss += loss.item()
        if i % 10 ==0:
            print(f'[{time_since(start)}] Epoch {epoch}', end='')
            print(f'[{i * len(inputs)}/{len(trainset)}]', end='')
            print(f'loss={total_loss / (i * len(inputs))}')
            
    return total_loss

def testModel():
    correct = 0
    total = len(testset)
    print("evaluating trained model...")
    with torch.no_grad():
        for i, (names, countries) in enumerate(testloader, 1):
            inputs, seq_lengths, target = make_tensors(names, countries) # 将名字的字符串转换成数字表示
            output = classifier(inputs, seq_lengths) # 预测输出
            pred = output.max(dim=1, keepdim=True)[1] # 预测出来是个向量，里面的值相当于概率，取最大的
            correct += pred.eq(target.view_as(pred)).sum().item() # 预测和实际标签相同则正确率加1
            
        percent = '%.2f' % (100 * correct / total)
        print(f'Test set:Accuracy{correct} / {total} {percent}%')
        
    return correct / total

def time_since(since):
    s = time.time() - since
    m = math.floor(s / 60)
    s -= m * 60
    return "%dm %ds" % (m, s)

if __name__ == "__main__":
    classifier = RNNClassifier(N_CHARS, HIDDEN_SIZE, N_COUNTRY, N_LAYER) # 定义模型
    if USE_GPU:
        device = torch.device("cuda:0")
        classifier.to(device)
    
    # 第三步：定义损失函数和优化器
    criterion = torch.nn.CrossEntropyLoss() # 分类问题使用交叉熵损失函数
    optimizer = torch.optim.Adam(classifier.parameters(), lr=0.001) # 使用了随机梯度下降法
    
    start = time.time()
    print("Training for %d epochs..." % N_EPOCHS)
    acc_list = []
    for epoch in range(1, N_EPOCHS + 1):
        # Train cycle
        trainModel()
        acc = testModel()
        acc_list.append(acc) # 存入列表，后面画图使用
        
    # 画图
    epoch = np.arange(1, len(acc_list) + 1, 1) # 步长为1
    acc_list = np.array(acc_list)
    plt.plot(epoch, acc_list)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.grid() # 显示网格线 1=True=默认显示；0=False=不显示
    plt.show()
