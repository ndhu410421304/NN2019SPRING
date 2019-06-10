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
start_time = time.time() #calculate time

#train data size
num = 27455

#setup
input_size = 1 #only one data-one label
hid_size1 = 64 #hidden layer size
hid_size2 = 128
hid_size3 = 256
hid_size4 = 512
hid_size5 = 1024 #adjust depend on convolution result
hid_size6 = 1024 #mlp hidden size
hid_size7 = 4096
num_classes = 25 #total gestures
num_epochs = 10 #10 epoch as 1 unit
batch_size = 1445 #1445 * 19
learning_rate = 0.00001 
test_size = 7172 #test data counts

torch.cuda.empty_cache() # for getting more memory for computation

acc = pd.read_csv('acc.csv').values #use highest result to compare
acclabel = acc[:,1:2] #take label
acclabel = acclabel.ravel() #to flat

ori_acc = 0 #for comparison

class train_dataset(data.Dataset): #read train data
	def __init__(self, filename):
		pd_data = pd.read_csv(filename).values #read csv
		self.data = pd_data[:,1:] #data
		self.data = np.reshape(self.data, (-1, 1, 28, 28)) #reshape to 4 dimension, (numitem) * 1 * 28*28(picture original shape)
		print(self.data.shape) #checking if correct
		self.label = pd_data[:,0:1] #label
		print(self.label.shape) #checking if correct
		self.length = self.data.shape[0]
    
	def __len__(self):
		return self.length # dataset class require
    
	def __getitem__(self, index):
		return torch.Tensor(self.data[index]), torch.Tensor(self.label[index]) # dataset class require

class test_dataset(data.Dataset): #read train data
	def __init__(self, filename):
		pd_data = pd.read_csv(filename).values #read csv
		self.data = pd_data[:,0:]
		self.data = np.reshape(self.data, (-1, 1, 28, 28))	#reshape to 4 dimension, (numitem) * 1 * 28*28(picture original shape)	
		self.length = self.data.shape[0]
    
	def __len__(self):
		return self.length # dataset class require
    
	def __getitem__(self, index):
		return torch.Tensor(self.data[index]) # dataset class require

traindata = train_dataset('train_data.csv') # create a item of class(which would read csv when initial)
testdata = test_dataset('test_data.csv')
trainloader = data.DataLoader(traindata, batch_size=batch_size,num_workers=0) #initial data loader by class item
testloader = data.DataLoader(testdata,batch_size=test_size,num_workers=0)
										  
class NetWork(nn.Module): #network structure
	def __init__(self, input_size, hid_size1, hid_size2, hid_size3, hid_size4, num_classes): # read size when initial calss member for more flexibility
		super(NetWork, self).__init__()
		self.conv1 = nn.Conv2d(input_size, hid_size1, kernel_size = 5, stride = 1, padding = 2) #convlution layer 1
		self.conv2 = nn.Conv2d(hid_size1, hid_size2, kernel_size = 3, padding = 2) #convlution layer 2
		self.conv3 = nn.Conv2d(hid_size2, hid_size3, kernel_size = 3, padding = 1) #convlution layer 3
		self.conv4 = nn.Conv2d(hid_size3, hid_size4, kernel_size = 3, padding = 1) #convlution layer 4
		self.conv5 = nn.Conv2d(hid_size4, hid_size5, kernel_size = 3, padding = 1) #convlution layer 5
		self.linear1 = nn.Linear(hid_size6, hid_size7) #mlp layer 1
		self.linear2 = nn.Linear(hid_size7, hid_size7) #mlp layer 2
		self.linear3 = nn.Linear(hid_size7, num_classes) #mlp layer 3
		self.pool = nn.MaxPool2d(3, 2) #maxpool
		self.norm = nn.BatchNorm2d(32) #batch nomarlization
		self.d = nn.Dropout(0.5) #drop 50 %
   
	def forward(self, x):
		hid_out1 = self.pool((F.relu(self.conv1(x)))) #cnn hidden output 1
		hid_out2 = self.pool((F.relu(self.conv2(hid_out1)))) #cnn hidden output 2
		hid_out3 = self.pool(F.relu(self.conv3(hid_out2))) #cnn hidden output 3
		hid_out4 = F.relu(self.conv4(hid_out3)) #cnn hidden output 5
		hid_out5 = self.pool((F.relu(self.conv5(hid_out4)))) #cnn hidden output 6
		fcinput1 = hid_out5.view(x.size(0), -1) #flatten cnn output for mlp layers
		lhid_out1 = self.linear1(F.relu(self.d(fcinput1))) #mlp hidden output1
		lhid_out2 = self.linear2(F.relu(self.d(lhid_out1))) #mlp hidden output2
		out = self.linear3(lhid_out2) #hidden output3
		prob = F.softmax(out, dim=1) #do softmax, then output
		return out

model2 = NetWork(input_size, hid_size1, hid_size2, hid_size3, hid_size4, num_classes).cuda() #initial network class member, using cuda to accelerate
criterion = nn.CrossEntropyLoss()  # use crossentropy for loss function
optimizer = torch.optim.Adam(model2.parameters(), lr=learning_rate) #se adam as optimizer

ep10 = 0 #count how many loop did we go till now

haccs = [] #hid accs, for observation
hloss = [] #hid loss, for observation
while True: #infinite loop
    for epoch in range(num_epochs): #0 ~9 run loop count depend on epoch
        for i, (data, label) in enumerate(trainloader): #batch input data from dataloader
            trainloss = []
			
            data = Variable(data.view(-1,1,28,28),requires_grad=False).cuda() #reshape data we get from data loader for cnn
            labels = Variable(label.long(),requires_grad=False).cuda() #use long to avoid RuntimeError: Expected object of scalar type Long but got scalar type Float for argument #2 'target'
            optimizer.zero_grad() #clear optimizer's grad
            outputs = model2(data) #get prediction from model
			
            labels = labels.view(outputs.shape[0]).cuda() #for batch size beside set ones

            loss = criterion(outputs, labels) #get this epoch's loss
            loss.backward() #back propagation
            optimizer.step() #optimizer optimize
			
            labelc = label.cuda() #duplicate label
			
            trainloss.append(loss.item()) #for calculate loss
			
            model2.eval() #evaluation mode
            correct = (outputs == labelc).sum() #calculate correct items

            if (i) % 18 == 0 and i != 0: #1445 * 19 -> loader load 19 times -> numbered as 0 - 18 => max is 18
                correct = torch.sum(torch.argmax(outputs,dim=1)==labels) # count the correct classification
                print ('Epoch: [%d/%d], Loss: %.4f, Accuracy: %.2f'
                        % (epoch+1 + 10*ep10, num_epochs + 10*ep10, np.mean(trainloss), (correct * 100 / outputs.shape[0]))) #for our observation

                with torch.no_grad(): # disable auto-grad
                    model3 = copy.deepcopy(model2) #copy model from current model and set it on CPU, for prevent to much thing in gpu
                    model3 = model3.cpu() #set on cpu
                    for i, (data) in enumerate(testloader): #load test data
                        data = Variable(data,requires_grad=False) #transfer format to Variable for network
                        
                        outputs = model3(data).cpu() #get prediction of test data
                        outputs = outputs.detach() #flat 
                        out, index = torch.max(outputs,1) #get predict label
                        indexc = index.cpu() #put in cpu
                        indexc = indexc.numpy() #change format prepare for observation and output
                        testy = indexc #copy
                        
                        dif = acclabel - testy #calculate how many item difference between our observation result and acquired output
                        hid_acc = (np.count_nonzero(dif == 0)/ 7172) #divide by total
                        print(hid_acc)
                        haccs.append(hid_acc) #recird local accuracy
                        hloss.append(trainloss) #record loss of this round
                        if(hid_acc > ori_acc): #get better accuracy = become better
                            print('Net saved.')
                            torch.save(model2, 'hid_net.pt') #save the netwotk
                            ori_acc = hid_acc #rest highest accuracy till now
                            output = pd.DataFrame({"samp_id": range(1,len(testy)+1), "label": testy}) #output result when we get current best
                            output.to_csv("nnhw1410421304h.csv", columns=["samp_id", "label"], index=False) #output result to csv
                        if (epoch) % 9 == 0 and (epoch) != 0: #last epoch in one loop
                            if ep10 % 100 == 0 and ep10 != 0: #1000th
                                #save per 1000
                                output = pd.DataFrame({"samp_id": range(1,len(testy)+1), "label": testy})
                                output.to_csv('nnhw1410421304(%d).csv'%(ep10 * 10 + epoch + 1 - 10), columns=["samp_id", "label"], index=False) #output result to csv
                                torch.save(model2, 'hid_net(%d).pt'%((ep10 * 10 + epoch + 1 - 10)))
                                #save data also
                                print(haccs)
                                haccsarray = np.asarray(haccs).ravel() #save accuracy
                                hlossarray = np.asarray(hloss).ravel() #save loss
                                output2 = pd.DataFrame({"samp_id": range(1,len(haccsarray)+1), "loss": hlossarray, "acc": haccsarray})
                                output2.to_csv('hidobs(%d).csv'%(ep10 * 10 + epoch + 1 - 10), columns=["samp_id", "loss", "acc"], index=False) #output result to csv for observatio
		
    ep10 = ep10 + 1 #set loop count + 1

#these part were originally use for output, but when we do infinite we dont need this	
model2 = model2.cpu() #put model on cpu when we are going to ouptu result to save moemory space

#basically same as output part in the infinite loop
with torch.no_grad(): # disable auto-grad
    for i, (data) in enumerate(testloader):
        data = Variable(data,requires_grad=False) #change data format
        
        outputs = model2(data) #get prediction
        outputs = outputs.detach() #flatten 
        out, index = torch.max(outputs,1)
        indexc = index.cpu() #put in cpu
        indexc = indexc.numpy() #change format for output
        testy = indexc
        
output = pd.DataFrame({"samp_id": range(1,len(testy)+1), "label": testy})
output.to_csv("nnhw1410421304.csv", columns=["samp_id", "label"], index=False) #output result to csv

torch.save(model2, 'net.pkl') #save model
print("--- %s seconds ---" % (time.time() - start_time)) #print total use time, originally use for observe performance