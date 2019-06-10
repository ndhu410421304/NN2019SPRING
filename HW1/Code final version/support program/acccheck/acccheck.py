import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision.transforms as transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np
import pandas as pd
import csv
import copy as copy
import time
start_time = time.time()

num = 27455
test_size = 7172
train_label = np.zeros((7172, 25))
csvcount = 24

guesslabel = np.zeros(7172)
print(guesslabel.shape)

acc = pd.read_csv('acc.csv').values #read observation usage prection data
testcsv = pd.read_csv('nnhw1410421304.csv').values #read our desire-submit data to check
acclabel = acc[:,1:2] #second row is label
testlabel = testcsv[:,1:2]

dif = acclabel - testlabel
print(np.count_nonzero(dif == 0)/ 7172) #show accuracy