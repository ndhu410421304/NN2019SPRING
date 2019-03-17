import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import csv

eta = 0.11 # could adjust this to see different result
traincsv = pd.read_csv('train_data.csv')
trainxarray = traincsv['x'].to_numpy()
trainyarray = traincsv['y'].to_numpy()
trainx = torch.from_numpy(trainxarray)
trainx = trainx.view(1,100) # reshape
trainy = torch.from_numpy(trainyarray)
trainy = trainy.view(1,100)

testcsv = pd.read_csv('test_data.csv')
testx = testcsv['x'].to_numpy()
testy = testcsv['x'].to_numpy()

'''
# Initialize w and b
'''
w1 = torch.randn((1, 1), requires_grad=True, dtype=torch.float64)
w2 = torch.randn((1, 1), requires_grad=True, dtype=torch.float64)
b = torch.randn((1, 1), requires_grad=True, dtype=torch.float64)
'''
# Iterative updating by gradient descent 
'''
for t in range(49): #could adjust time looping

    y_pred = (w1.mm(trainx.pow(2))).add(w2.mm(trainx)).add(b) # predict y using w1, w2 and b
    L = ((y_pred-trainy)*(y_pred-trainy)).mean() # calculate L
    L.backward() # calculate gradients

    with torch.no_grad(): # disable autograd
        w1 -= eta * w1.grad
        w2 -= eta * w2.grad
        b -= eta * b.grad
        w1.grad.zero_() # clear accumulated gradients
        w2.grad.zero_()
        b.grad.zero_()
        
for t in range(99):
    testy[t] = w1*testx[t]*testx[t]+w2*testx[t]+b

output = pd.DataFrame({"Id": range(1,len(testy)+1), "Expected": testy})
output.to_csv("nnbonus1410421304.csv", columns=["Id", "Expected"], index=False)

print(w1) #check result
print(w2)
print(b)