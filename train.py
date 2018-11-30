import torch
import torch.autograd
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable
from torch import optim
import numpy as np
import math, random
import arglist
import util
import os

use_cuda = arglist.use_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

class ClassificationModel(nn.Module):
    def __init__(self):
        super(ClassificationModel, self).__init__()
        self.fc1 = nn.Linear(arglist.nn_input_size, arglist.nn_hidden_size)
        self.fc2 = nn.Linear(arglist.nn_hidden_size, arglist.nn_hidden_size)
        self.fc3 = nn.Linear(arglist.nn_hidden_size, arglist.nn_output_size)

    def forward(self, i):
        h = F.relu(self.fc1(i))
        h = F.relu(self.fc2(h))
        o = F.softmax(self.fc3(h))
        return o

def build_network():
    model = ClassificationModel().to(device)
    return model

def load_dataset(train_mode=True):
    if train_mode:
        d = util.load_pickle(arglist.pickle_file, arglist.pickle_dir + os.path.sep + "train")
        return d
    else:
        d = util.load_pickle(arglist.pickle_file, arglist.pickle_dir + os.path.sep + "test")
        return d

def rolling(df, window, step):
    count = 0
    df_length = len(df)
    while count < (df_length - window):
        yield count, df[count:window+count]
        count+=step

d_train = load_dataset(train_mode=True)
model = build_network()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# This is particularly useful when you have an unbalanced training set.
criterion = nn.CrossEntropyLoss()

ave_loss = 0

for epoch in range(arglist.n_epochs):

    for batch_idx, data in rolling(d_train, arglist.n_batch_size, arglist.n_batch_size):
        optimizer.zero_grad()

        """
        temporarily, 30 -> 10
        """
        change30 = lambda x: 10 if x == 30 else x
        x = Variable(torch.from_numpy(data.drop(['y', 'map', 'race'], axis=1).values).float())
        target = Variable(torch.from_numpy(np.asarray([change30(x) for x in data['y'].values])).long())

        if arglist.use_cuda:
            print('use cuda')
            x, target = x.cuda(), y.cuda()

        predicted = model(x)
        loss = criterion(predicted, target)
        ave_loss = ave_loss * 0.9 + loss.data[0] * 0.1
        loss.backward()
        optimizer.step()

        # if (batch_idx) % 20 == 0 or (batch_idx + 1) == len(d_train):
        print('==>>> epoch: {}, batch index: {}, train loss: {:.6f}'.format(
            epoch, batch_idx + 1, ave_loss))

        correct_cnt, ave_loss = 0, 0
        total_cnt = 0


