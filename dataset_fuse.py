from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class MyDataset_v1(Dataset):
    def __init__(self):
        self.data = [1, 2, 3, 4]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]


class MyDataset_v2(Dataset):
    def __init__(self):
        self.data = [1.1, 2.2, 3.3, 4.4]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]


class MyDataset(Dataset):
    def __init__(self, dataset1, dataset2):
        self.dataset1 = dataset1
        self.dataset2 = dataset2

    def __getitem__(self, index):
        x1 = self.dataset1[index]
        x2 = self.dataset2[index]
        return x1, x2

    def __len__(self):
        return len(self.dataset1)


if __name__ == "__main__":
    myDataset1 = MyDataset_v1()
    myDataset2 = MyDataset_v2()

    myDataset = MyDataset(dataset1=myDataset1, dataset2=myDataset2)
    dataloader = DataLoader(dataset=myDataset, batch_size=2, shuffle=True, pin_memory=True)
    epoch = 2
    step = -1
    for i in range(epoch):
        for batch_ind, data in enumerate(dataloader):
            data1, data2 = data[0], data[1]
            print("Epoch: {} Batch_ind: {} data in Dataset1: {} data in Dataset2: {}".format(i, batch_ind, data1, data2))
