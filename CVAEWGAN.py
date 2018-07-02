from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
import torchvision.utils as vutils

class Encode(nn.Module):
    def __init__(self):
        super(Encode, self).__init__()

        self.fc1 = nn.Linear(784, 400)
        self.fc21 = nn.Linear(400, 20)
        self.fc22 = nn.Linear(400, 20)
        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, 784)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        self.mu_ = nn.Sequential(
            #28x28->12x12
            nn.Conv2d(1,8,5,2,0,bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            #12x12->4x4
            nn.Conv2d(8,64,5,2,0,bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            #4x4->1x1: 20,1,1
            nn.Conv2d(64,20,4,1,0,bias=False),
            nn.ReLU(True)
            )


        self.dec_ = nn.Sequential(
            #1x1->4x4
            nn.ConvTranspose2d(30,20*8,4,1,0,bias=False),  #(ic,oc,kernel,stride,padding)
            nn.BatchNorm2d(20*8), 
            nn.ReLU(True),
            nn.ConvTranspose2d(20*8,20*16,4,2,1,bias=False), #4x4->8x8
            nn.BatchNorm2d(20*16),
            nn.ReLU(True),
            nn.ConvTranspose2d(20*16,20*32,4,2,1,bias=False), #8x8->16x16
            nn.BatchNorm2d(20*32),
            nn.ReLU(True),
            nn.ConvTranspose2d(20*32,1,2,2,2,bias=False), #16x16->28x28
            nn.Sigmoid()
            )
        

    def encode(self, x):
        h1 = self.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def encode_cnn(self,x):
        return self.mu_(x)

    def reparameterize(self, mu, logvar):
        if self.training:
          std = logvar.mul(0.5).exp_()
          eps = Variable(std.data.new(std.size()).normal_())
          return eps.mul(std).add_(mu)
        else:
          return mu


    def decode_new(self,z):
        z = z.view(-1,z.size(1),1,1)
        return(self.dec_(z))
        
      
    def decode(self, z):
        z = z.view(-1,20)
        h3 = self.relu(self.fc3(z))
        return self.sigmoid(self.fc4(h3))

    def dec_params(self):
        return self.fc3, self.fc4

    def return_weights(self):
        return self.fc3.weight, self.fc4.weight

    def forward(self, x):
       mu = self.encode_cnn(x.view(-1, 1,28,28))
       return mu


class Decode(nn.Module):
    def __init__(self):
        super(Decode,self).__init__()

        self.dec_ = nn.Sequential(
            #1x1->4x4
            nn.ConvTranspose2d(30,20*8,4,1,0,bias=False),  #(ic,oc,kernel,stride,padding)
            nn.BatchNorm2d(20*8), 
            nn.ReLU(True),
            nn.ConvTranspose2d(20*8,20*16,4,2,1,bias=False), #4x4->8x8
            nn.BatchNorm2d(20*16),
            nn.ReLU(True),
            nn.ConvTranspose2d(20*16,20*32,4,2,1,bias=False), #8x8->16x16
            nn.BatchNorm2d(20*32),
            nn.ReLU(True),
            nn.ConvTranspose2d(20*32,1,2,2,2,bias=False), #16x16->28x28
            nn.Sigmoid()
            )

    def forward(self,z,label):
        label = label.unsqueeze(2).unsqueeze(2)
        z = torch.cat((z,label),1)
        z = z.view(-1,z.size(1),1,1)
        return(self.dec_(z))





class NetD(nn.Module):
    def __init__(self,n_label,n_z,dim_h):
        super(NetD,self).__init__()

        
        self.n_label = n_label
        self.dim_h = dim_h
        self.n_z = n_z

        self.main = nn.Sequential(
            nn.Linear(self.n_z+self.n_label, self.dim_h * 4),
            nn.ReLU(True),
            nn.Linear(self.dim_h * 4, self.dim_h * 4),
            nn.ReLU(True),
            nn.Linear(self.dim_h * 4, self.dim_h * 4),
            nn.ReLU(True),
            nn.Linear(self.dim_h * 4, self.dim_h * 4),
            nn.ReLU(True),
            nn.Linear(self.dim_h * 4, 1),
            nn.Sigmoid()
        )

    def forward(self,z,label):
        z = z.squeeze(2).squeeze(2)
        z = torch.cat((z,label),1)
        x = self.main(z)
        return x


