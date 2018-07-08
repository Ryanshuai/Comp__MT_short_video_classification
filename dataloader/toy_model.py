import torch
import torch.nn as nn
import numpy as np
import torchvision.transforms.functional as TF

class CNN_Model(nn.Module):
    
    def __init__(self,in_channels=3):
        
        super().__init__()
        self.in_channels = in_channels
        self.cnn_layers = nn.Sequential(
            nn.Conv3d(in_channels,8,kernel_size=(3,3,1),padding=(1,1,0)),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2,2,1),stride=(2,2,1)),
            nn.Conv3d(8,8,kernel_size=(3,3,1),padding=(1,1,0)),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2,2,1),stride=(2,2,1)),
            nn.Conv3d(8,8,kernel_size=(3,3,1),padding=(1,1,0)),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2,2,1),stride=(2,2,1)),
            nn.Conv3d(8,8,kernel_size=(3,3,1),padding=(1,1,0)),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2,2,1),stride=(2,2,1)),
            nn.Conv3d(8,8,kernel_size=(3,3,1),padding=(1,1,0)),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2,2,1),stride=(2,2,1)))
    
    def forward(self,x):
        
        assert list(x.shape)[2:4] == [64,64]
        
        x = self.cnn_layers(x)
        x = x.view(-1,8*2*2,list(x.shape)[4])
        x = torch.transpose(x,1,2)
        x = torch.transpose(x,0,1)
        
        return x


class RNN_Model(nn.Module):
    
    def __init__(self,input_size,hidden_size=64):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.LSTM = nn.LSTM(input_size,hidden_size)
    
    def forward(self,x):
        
        assert list(x.shape)[2] == self.input_size
        
        x = self.LSTM(x)
        
        return x

class Classifier(nn.Module):
    
    def __init__(self,input_size,output_size):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.clf = nn.Sequential(
            nn.Linear(input_size,64),
            nn.ReLU(),
            nn.Linear(64,output_size))
    def forward(self,x):
        
        assert list(x.shape)[1] == self.input_size
        
        x = self.clf(x)
        
        return x

class Toy_Model(nn.Module):
    
    def __init__(self,output_size):
        super().__init__()
        self.output_size = output_size
        self.cnn = CNN_Model()
        self.rnn = RNN_Model(input_size=32,hidden_size=32)
        self.clf = Classifier(input_size=32,output_size=output_size)
        
    def forward(self,x):
        
        x = self.cnn(x)
        #x = self.vlad(x)
        x = self.rnn(x)[0][-1]
        x = self.clf(x)
        
        return x