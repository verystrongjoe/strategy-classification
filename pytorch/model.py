import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import MNIST
import arglist
import util
import os
import torch.utils.data as td
import torchvision.datasets
import torchvision.transforms as transforms
import torch.optim as optim
from torch.autograd import Variable
import multiprocessing

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution kernel
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)

        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(256, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.name = 'TEST'

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

if __name__ == '__main__':
    multiprocessing.freeze_support()

    trans = transforms.Compose([transforms.ToTensor()])

    d_train = MNIST(arglist.mnist_data, train=True, download=True, transform=trans)
    d_test  = MNIST(arglist.mnist_data, train=False, download=True, transform=trans)

    train_loader = td.DataLoader(d_train, batch_size=128, shuffle=True, num_workers=arglist.n_threads)
    test_loader = td.DataLoader(d_test, batch_size=128, shuffle=True, num_workers=arglist.n_threads)

    ## training
    model = Net()

    if arglist.use_cuda:
        model = model.cuda()

    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(arglist.n_epochs):

        # training
        ave_loss = 0

        for batch_idx, (x, target) in enumerate(train_loader):

            optimizer.zero_grad()

            if arglist.use_cuda:
                x, target = x.cuda(), target.cuda()
            x, target = Variable(x), Variable(target)
            out = model(x)
            loss = criterion(out, target)
            ave_loss = ave_loss * 0.9 + loss.data[0] * 0.1
            loss.backward()
            optimizer.step()

            if (batch_idx + 1) % 100 == 0 or (batch_idx + 1) == len(train_loader):
                print('==>>> epoch: {}, batch index: {}, train loss: {:.6f}'.format(
                    epoch, batch_idx + 1, ave_loss))

            correct_cnt, ave_loss = 0, 0
            total_cnt = 0

        for batch_idx, (x, target) in enumerate(test_loader):
            if arglist.use_cuda:
                x, target = x.cuda(), target.cuda()
            x, target = Variable(x, volatile=True), Variable(target, volatile=True)
            out = model(x)
            loss = criterion(out, target)
            _, pred_label = torch.max(out.data, 1)
            total_cnt += x.data.size()[0]
            correct_cnt += (pred_label == target.data).sum()
            # smooth average
            ave_loss = ave_loss * 0.9 + loss.data[0] * 0.1

            if (batch_idx + 1) % 100 == 0 or (batch_idx + 1) == len(test_loader):
                print('==>>> epoch: {}, batch index: {}, test loss: {:.6f}, acc: {:.3f}'.format(
                    epoch, batch_idx + 1, ave_loss, correct_cnt * 1.0 / total_cnt))

       # torch.save(model.state_dict(), model.name())