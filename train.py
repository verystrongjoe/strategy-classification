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
        o = self.fc3(h)
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
d_test = load_dataset(train_mode=False)
model = build_network()


# https://stackoverflow.com/questions/51102205/how-to-know-the-labels-assigned-by-astypecategory-cat-codes
categories_counts = d_train['y'].astype('category').cat.codes.value_counts()
categories = d_train['y'].astype('category').cat.codes.unique()


print('--------------------------------------------')
print('total count :{}'.format(len(d_train)))
print('--------------------------------------------')
print('counts of each strategies')
print('--------------------------------------------')

for c in categories:
    print(' origin {}, category code : {}, counts {}'.format(
        d_train['y'].astype('category').cat.categories[c],
        c, categories_counts[c]
    )
    )
print('--------------------------------------------')


# This is particularly useful when you have an unbalanced training set.
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.02, momentum=0.9)


ave_loss = 0

for epoch in range(arglist.n_epochs):

    # training
    running_loss = 0

    for batch_idx, data in rolling(d_train, arglist.n_batch_size, arglist.n_batch_size):
        """
        temporarily, 30 -> 10
        """
        change30 = lambda x: 10 if x == 30 else x
        x = Variable(torch.from_numpy(data.drop(['y', 'map', 'race', 'time'], axis=1).values).float())
        target = Variable(torch.from_numpy(np.asarray([change30(x) for x in data['y'].values])).long())

        if arglist.use_cuda:
            print('use cuda')
            x, target = x.cuda(), target.cuda()

        optimizer.zero_grad()

        predicted = model(x)
        loss = criterion(predicted, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        # print(loss.item())
        # if (batch_idx) % 1000 == 0:
    print('==>>> epoch: {}, train loss: {:.6f}'.format(epoch, running_loss))


"""
model test phrase
"""

model.train(False)

n_correct = 0

for batch_idx, data in rolling(d_test, arglist.n_batch_size, arglist.n_batch_size):

    """
    temporarily, 30 -> 10
    """
    change30 = lambda x: 10 if x == 30 else x
    x = Variable(torch.from_numpy(data.drop(['y', 'map', 'race', 'time'], axis=1).values).float())
    # target = Variable(torch.from_numpy(np.asarray([change30(x) for x in data['y'].values])).long())
    target = np.asarray([change30(x) for x in data['y'].values])

    if arglist.use_cuda:
        print('use cuda')
        x, target = x.cuda(), target.cuda()

    predicted = model(x)
    y = predicted.detach().numpy()
    y = np.argmax(y, axis=1)

    n_correct = n_correct + np.sum(target==y)

print('accuracy : {}'.format(n_correct / len(d_test)))