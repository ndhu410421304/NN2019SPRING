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

traincsv = pd.read_csv('train_data.csv')

trainx1array = traincsv['x1'].to_numpy()
trainx2array = traincsv['x2'].to_numpy()
trainyarray = traincsv['label'].to_numpy()
trainx1 = torch.from_numpy(trainx1array)
trainx2 = torch.from_numpy(trainx2array)
trainy = torch.from_numpy(trainyarray)

testcsv = pd.read_csv('test_data.csv')
testx1array = testcsv['x1'].to_numpy()
testx2array = testcsv['x2'].to_numpy()
testx1 = torch.from_numpy(testx1array)
testx2 = torch.from_numpy(testx2array)
testy = testcsv['x1'].to_numpy()

input_size = 2
hid_size1 = 20
hid_size2 = 20
hid_size3 = 20
num_classes = 3
num_epochs = 250
batch_size = 300
learning_rate = 0.01

class train_dataset(data.Dataset):
	def __init__(self, filename):
		pd_data = pd.read_csv(filename).values
		self.data = pd_data[:,0:2]
		self.label = pd_data[:,2:]
		self.length = self.data.shape[0]
    
	def __len__(self):
		return self.length
    
	def __getitem__(self, index):
		return torch.Tensor(self.data[index]), torch.Tensor(self.label[index])

class test_dataset(data.Dataset):
	def __init__(self, filename):
		pd_data = pd.read_csv(filename).values
		self.data = pd_data[:,0:2] 
		self.length = self.data.shape[0]
    
	def __len__(self):
		return self.length
    
	def __getitem__(self, index):
		return torch.Tensor(self.data[index])

traindata = train_dataset('train_data.csv')

testdata = test_dataset('test_data.csv')
testloader = data.DataLoader(testdata,batch_size=300,num_workers=0)

trainsplitsize = int(0.75 * len(traindata))
validsplitsize = len(traindata) - trainsplitsize
trainsplit, validsplit = torch.utils.data.random_split(traindata, [trainsplitsize, validsplitsize])

trainsplitloader = data.DataLoader(trainsplit, batch_size=300)
validsplitloader = data.DataLoader(validsplit, batch_size=300)
										  
class NetWork(nn.Module):
	def __init__(self, input_size, hid_size1, hid_size2, hid_size3, num_classes):
		super(NetWork, self).__init__()
		self.d = nn.Dropout(p=0.1)
		self.linear1 = nn.Linear(input_size, hid_size1)
		self.linear2 = nn.Linear(hid_size1, hid_size2)
		self.linear3 = nn.Linear(hid_size2, hid_size3)
		self.linear4 = nn.Linear(hid_size3, num_classes)
    
	def forward(self, x):
		hid_out1 = F.relu(self.linear1(x))
		hid_out1 = self.d(hid_out1)
		hid_out2 = F.relu(self.linear2(hid_out1))
		hid_out2 = self.d(hid_out2)
		hid_out3 = F.relu(self.linear3(hid_out2))
		hid_out3 = self.d(hid_out3)
		out = self.linear4(hid_out3)
		prob = F.softmax(out, dim=1)
		return out

model2 = NetWork(input_size, hid_size1, hid_size2, hid_size3, num_classes)
criterion = nn.CrossEntropyLoss() 
optimizer = torch.optim.Adam(model2.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
	for i, (data, label) in enumerate(trainsplitloader):
		
		trainloss, validloss = [], []
		
		data = Variable(data,requires_grad=False)
		labels = Variable(label.long(),requires_grad=False)
		
		optimizer.zero_grad()
		outputs = model2(data)
		loss = criterion(outputs,labels.view(-1))
		loss.backward()
		optimizer.step()
		
		trainloss.append(loss.item())
		
		model2.eval()
		for data, label in validsplitloader:
			outputs = model2(data)
			loss = criterion(outputs,labels.view(-1))
			validloss.append(loss.item())
		if (i) % 3 == 0:
			correct = torch.sum(torch.argmax(outputs,dim=1)==labels) # count the correct classification
			print ('Epoch: [%d/%d], Loss: %.4f, Valid_Loss: %.4f Accuracy: %.2f'
					% (epoch+1, num_epochs, np.mean(trainloss), np.mean(validloss), (correct.item())/batch_size))

with torch.no_grad(): # disable auto-grad
	for i, (data) in enumerate(testloader):
		data = Variable(data,requires_grad=False)
		
		outputs = model2(data)
		outputs = outputs.exp().detach() #need exp?
		out, index = torch.max(outputs,1)
		index = index.numpy()
		testy = index
		
output = pd.DataFrame({"id": range(1,len(testy)+1), "predicted label": testy})
output.to_csv("nnbonus2410421304.csv", columns=["id", "predicted label"], index=False) #output result to csv

torch.save(model2, 'net.pkl')