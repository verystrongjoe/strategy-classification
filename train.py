import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable
from torch import optim
import numpy as np
import math, random
import arglist
import util

use_cuda = not arglist.no_cuda and torch.cuda.is_available()
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


def load_dataset():
    d = util.load_pickle(arglist.pickle_file, arglist.pickle_dir)
    return d


d = load_dataset()
model = build_network()

loss = nn.MSELoss()
optim = optim.SGD(model.parameters(), lr=1e-3)

losses = np.zeros(arglist.n_epochs) # For plotting

"""
TBD
"""

