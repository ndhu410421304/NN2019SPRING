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

#setup
input_size = 2 #x1 and x2
hid_size1 = 36 #hidden layer size
hid_size2 = 84
hid_size3 = 36
num_classes = 3 #labels count
num_epochs = 150 #number of epochs
batch_size = 300 #minibatch size
learning_rate = 0.001

#class for dataloader constructor
class train_dataset(data.Dataset): #inherit dataset class
	def __init__(self, filename):
		traindata = pd.read_csv(filename).values
		self.data = traindata[:,0:2] #get x1 and x2
		self.label = traindata[:,2:] #ggt label
		self.length = self.data.shape[0]
    
	def __len__(self):
		return self.length
    
	def __getitem__(self, index):
		return torch.Tensor(self.data[index]), torch.Tensor(self.label[index])

class test_dataset(data.Dataset):
	def __init__(self, filename):
		testdata = pd.read_csv(filename).values
		self.data = testdata[:,0:2] #only need to deal with data
		self.length = self.data.shape[0]
    
	def __len__(self):
		return self.length
    
	def __getitem__(self, index):
		return torch.Tensor(self.data[index])

traindata = train_dataset('train_data.csv') #create class member of dataset
testdata = test_dataset('test_data.csv')

trainloader = data.DataLoader(traindata, batch_size=batch_size, num_workers=0) #train, test loader with minibatch
testloader = data.DataLoader(testdata,batch_size=batch_size, num_workers=0)
plotloader = data.DataLoader(traindata, batch_size=1200, num_workers=0) #plotloader for plot train result

#network setup
class NetWork(nn.Module):
	def __init__(self, input_size, hid_size1, hid_size2, hid_size3, num_classes):
		super(NetWork, self).__init__()
		self.d = nn.Dropout(p=0.5) #50%->0
		self.linear1 = nn.Linear(input_size, hid_size1) #input to first hidden layer
		self.linear2 = nn.Linear(hid_size1, hid_size2) #to second
		self.linear3 = nn.Linear(hid_size2, hid_size3) #to third
		self.out = nn.Linear(hid_size3, num_classes) #to output layer
		self.b1 = nn.BatchNorm1d(hid_size1) #normalization
		self.b2 = nn.BatchNorm1d(hid_size2)
		self.b3 = nn.BatchNorm1d(hid_size3)
   
	def forward(self, x):
		hid_out1 = self.d(self.b1(F.relu(self.linear1(x)))) #hidden output1
		hid_out2 = self.d(self.b2(F.relu(self.linear2(hid_out1)))) #hidden output1
		hid_out3 = self.d(self.b3(F.relu(self.linear3(hid_out2)))) #hidden output1
		out = self.out(hid_out3)
		prob = F.softmax(out, dim=1)
		return out

#create model and loss function, optimizer
model2 = NetWork(input_size, hid_size1, hid_size2, hid_size3, num_classes)
criterion = nn.CrossEntropyLoss()  #loss function
optimizer = torch.optim.Adam(model2.parameters(), lr=learning_rate) #optimizer

minloss = 0 #use to record minimum loss
minmodel = model2 #use to save minimum model

for epoch in range(num_epochs): #loop for epocch times
	for i, (data, label) in enumerate(trainloader):
		
		trainloss = []
		
		data = Variable(data,requires_grad=False) #change datatype
		labels = Variable(label.long(),requires_grad=False)
		optimizer.zero_grad()
		outputs = model2(data) #get the prediction
		loss = criterion(outputs,labels.view(-1)) #caculate loss
		loss.backward() #backpropagation
		optimizer.step() #activae activation function
		trainloss.append(loss.item())
		model2.eval()

		if (i) % 6 == 0:
			correct = torch.sum(torch.argmax(outputs,dim=1)==labels) #output training result
			print ('Epoch: [%d/%d], Loss: %.4f, Accuracy: %.2f'
					% (epoch+1, num_epochs, np.mean(trainloss), (correct / outputs.shape[0])))
					
		if minloss == 0: #first epoch
			minloss = np.mean(trainloss)
			minmodel = copy.deepcopy(model2)
		else: #save model for minium loss
			if minloss > np.mean(trainloss):
				minloss = np.mean(trainloss)
				minmodel = copy.deepcopy(model2)
					
model2 = copy.deepcopy(minmodel) #change model to minimum loss one
print('%.4f'%(minloss)) #minimum loss
print('%.4f'%(np.mean(trainloss))) #current loss

plt.figure(num = 'Train Result') #setup window title
plt.clf()

#plot traing result
with torch.no_grad(): # disable auto-grad	
	for i, (data, label) in enumerate(plotloader): #plot training result
		data = Variable(data,requires_grad=False)
		labels = Variable(label.long(),requires_grad=False)
		outputs = model2(data)
		outputs = outputs.detach()
		out, index = torch.max(outputs,1)
		index = index.numpy()
		
		colors = ['r','y','b'] #three color
		datan = data.numpy()
		labeln = label.numpy()
		
		for j in range(3): #plot input data with label
			plotdata = datan[labeln[:,0] == j,:] #when label = choosen label -> choosen color
			plt.scatter(plotdata[:,0],plotdata[:,1],facecolors=colors[j]) #x1,x2, with its color

		for k in range(3): #plot prediction
			plotdata = datan[index==k,:] #when prediction = choosen label -> choosen color
			plt.scatter(plotdata[:,0],plotdata[:,1],edgecolors=colors[k],facecolors='none') #x1,x2, with its color
		
		plt.show()
		
plt.figure(num = 'Test Result')	#setup window title	
plt.clf()

#get prediction of test case and plot he prediction
with torch.no_grad(): # disable auto-grad
	for i, (data) in enumerate(testloader):
		data = Variable(data,requires_grad=False)
		
		outputs = model2(data) #get result
		outputs = outputs.detach()
		out, index = torch.max(outputs,1) # choose most familiar one
		index = index.numpy()
		testy = index #use for output
		
		colors = ['r','y','b'] #three color
		datan = data.numpy()
		
		for j in range(3): #plot prediction
			plotdata = datan[index==j,:] #when prediction = choosen label -> choosen color
			plt.scatter(plotdata[:,0],plotdata[:,1],edgecolors=colors[j],facecolors='none') #x1,x2, with its color
		
		plt.show()
		
output = pd.DataFrame({"id": range(1,len(testy)+1), "predicted label": testy})
output.to_csv("nnbonus2410421304.csv", columns=["id", "predicted label"], index=False) #output result to csv

torch.save(model2, 'net.pkl') #save model