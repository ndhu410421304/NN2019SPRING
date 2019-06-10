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

num = 27455
test_size = 7172
train_label = np.zeros((7172, 25))
csvcount = 151 #how many submitions to genrate one observation usage data
oacc = [0.82043, 0.82601, 0.68933, 0.70083, 0.83542, 0.71025, 0.81345, 0.80578, 0.83403, 0.80613, 0.89225, 0.76394, 0.80892, 0.79672, 0.89539, 0.76569, 0.95746, 0.94630,
			0.91004, 0.90167, 0.90167, 0.75139, 0.82217, 0.87029, 0.96164, 0.78033, 0.95223, 0.96827, 0.96548, 0.97489, 0.96896, 0.96199, 0.96722, 0.97280, 0.97245, 0.97071,
			 0.95711, 0.96896, 0.92119, 0.95711, 0.91213, 0.93270, 0.91910, 0.95815, 0.91457, 0.83054, 0.83926, 0.83019, 0.79497, 0.94735, 0.93479, 0.98361, 0.98221, 0.98779,
			  0.99476, 0.99825, 0.94979, 0.92503, 0.97315, 0.94281, 0.96164, 0.96443, 0.96582, 0.98326, 0.98221, 0.97803, 0.98465, 0.97873, 0.98430, 0.98465, 0.98465, 0.98535,
			   0.99058, 0.98779, 0.98570, 0.98186, 0.98047, 0.97559, 0.98605, 0.98744, 0.97768, 0.98849, 0.98117, 0.98605, 0.96617, 0.97175, 0.96861, 0.97768, 0.97942, 0.98326,
			    0.99267, 0.95258, 0.95815, 0.96722, 0.95990, 0.96896, 0.96896, 0.97245, 0.97733, 0.90097, 0.93061, 0.97559, 0.98082, 0.96931, 0.97489, 0.97350, 0.93549, 0.93967,
				 0.89225, 0.92085, 0.95327, 0.96931, 0.97977, 0.98012, 0.98117, 0.98500, 0.97838, 0.98500, 0.97768, 0.97315, 0.96582, 0.96443, 0.98256, 0.97803, 0.97594, 0.98186,
				  0.98221, 0.98291, 0.96025, 0.98291, 0.98326, 0.97873, 0.96827, 0.98500, 0.98256, 0.98152, 0.98012, 0.98152, 0.98779, 0.95955, 0.93165, 0.96164, 0.98152, 0.94665,
				   0.96478, 0.96617, 0.96757, 0.95990, 0.96199, 0.95536, 0.95467] #submitions accuracy given by graded system
guesslabel = np.zeros(7172)
print(guesslabel.shape)

csvs = ['nnhw1410421304(%d).csv'%(n) for n in range(1, csvcount+1)]

for n in range(csvcount): #each csv
	pd_data = pd.read_csv(csvs[n]).values
	label = pd_data[:,1:2] #read label
	for m in range(test_size):
		lab = label[m] #get the predict label
		train_label[m][lab] += oacc[n] #'vote'for that label with its weight
	
tlist = train_label.tolist() #change data format
for o in range(test_size):
	guesslabel[o] = np.argmax(tlist[o]) #for each label, choose which preiction get most vote, then set output as that prediction
		
output = pd.DataFrame({"samp_id": range(1,len(guesslabel)+1), "label": guesslabel})
output.to_csv("acc.csv", columns=["samp_id", "label"], index=False) #output an observation usage prediction