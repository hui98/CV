import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import UT_PIL_Torchsave as SBLoader
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.optim as optim
import time
from torchvision import datasets, transforms


class PixelBlock(nn.Module):
    def __init__(self,inc,outc,scalesize,kernel,stride,padding):
        super(PixelBlock,self).__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(inc,outc,kernel,stride,padding),
            nn.PixelShuffle(scalesize),
            nn.BatchNorm2d(inc),
            nn.ReLU(inplace=True),
        )
    def forward(self,input):
        out = self.seq(input)
        return out
class ResBlock(nn.Module):
    def __init__(self,channel,kernel,stride,padding):
        super(ResBlock,self).__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(channel,channel,kernel,stride,padding),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel,channel,kernel,stride,padding),
            nn.BatchNorm2d(channel),
        )
        
    def forward(self,input):
        dout = self.seq(input)
        mixout = input + dout
        return mixout
class Generator(nn.Module):
    def __init__(self,kinds = 3,resblocknum = 16):
        super(Generator,self).__init__()
        self.linear1 = nn.Linear(100,64*16*16)
        self.BN1 = nn.BatchNorm2d(64)
        self.RBlis = []
        for i in range(0,resblocknum):
            rs = ResBlock(64,3,1,1)
            self.RBlis.append(rs)
            self.add_module('res'+str(i),rs)
            
        self.seq2 = nn.Sequential(
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.seq3 = nn.Sequential(
            PixelBlock(64,256,2,3,1,1),
            PixelBlock(64,256,2,3,1,1),
            PixelBlock(64,256,2,3,1,1),
        )
        self.convf = nn.Conv2d(64,3,9,1,4)
    def forward(self,z,l):
        vec1 = self.linear1(z)
        vec1 = F.relu(self.BN1(vec1.view(-1,64,16,16)))
        out1 = vec1
        for i in self.RBlis:
            vec1 = i(vec1)
        vec1 = self.seq2(vec1)
        out1 = vec1 + out1
        out = torch.tanh(self.convf(self.seq3(out1)))
        return out
        
class ResBlock_1(nn.Module):
    def __init__(self):
        super(ResBlock_1,self).__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(32,32,3,1,1),
            #nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1, inplace=False),
            nn.Conv2d(32,32,3,1,1),
            #nn.BatchNorm2d(32),
        )
        self.lr = nn.LeakyReLU(0.1, inplace=False)
    
    def forward(self,input):
        out1 = self.seq(input) + input
        out = self.lr(out1)
        return out

class ResBlock_2(nn.Module):
    def __init__(self,channel,kernel,stride,padding):
        super(ResBlock_2,self).__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(channel,channel,kernel,stride,padding),
            #nn.BatchNorm2d(channel),
            nn.LeakyReLU(0.1, inplace=False),
            nn.Conv2d(channel,channel,kernel,stride,padding),
            #nn.BatchNorm2d(channel),
            nn.LeakyReLU(0.1, inplace=False),
        )
        self.lr = nn.LeakyReLU(0.1, inplace=False)
    def forward(self,input):
        out1 = self.seq(input) + input
        out1 = self.seq(input)
        out = self.lr(out1)
        return out
            

class Discriminator111(nn.Module):
    def __init__(self,kinds = 3):
        super(Discriminator,self).__init__()
        self.kinds = kinds
        self.seq1 = nn.Sequential(
            nn.Conv2d(3,32,4,2,1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1, inplace=False),
        )
        self.seq2 = nn.Sequential(
            #ResBlock_1(),
            ResBlock_1(),
            nn.Conv2d(32,64,4,2,1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=False),
            #ResBlock_2(64,3,1,1),
            #ResBlock_2(64,3,1,1),
            #ResBlock_2(64,3,1,1),
            ResBlock_2(64,3,1,1),
            nn.Conv2d(64,128,4,2,1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=False),
            #ResBlock_2(128,3,1,1),
            #ResBlock_2(128,3,1,1),
            #ResBlock_2(128,3,1,1),
            ResBlock_2(128,3,1,1),
            nn.Conv2d(128,256,4,2,1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=False),
            #ResBlock_2(256,3,1,1),
            #ResBlock_2(256,3,1,1),
            #ResBlock_2(256,3,1,1),
            ResBlock_2(256,3,1,1),
            nn.Conv2d(256,512,4,2,1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=False),
            #ResBlock_2(512,3,1,1),
            #ResBlock_2(512,3,1,1),
            #ResBlock_2(512,3,1,1),
            ResBlock_2(512,3,1,1),
            nn.Conv2d(512,1024,4,2,1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Conv2d(1024,1,4,2,1),
        )
    
    def forward(self,imgs,labels):
        out1 = self.seq1(imgs)
        out2 = self.seq2(out1)
        out2 = out2.view(-1,1)
        return out2
        
        
class Discriminator(nn.Module):
    def __init__(self,kinds = 3):
        super(Discriminator,self).__init__()
        self.kinds = kinds
        self.embed = nn.Embedding(self.kinds,5*32*32)
        self.conv1 = nn.Conv2d(3,20,6,2,2)
        self.BN1 = nn.BatchNorm2d(20)
        self.conv2 = nn.Conv2d(20,10,4,2,1)
        self.BN2 = nn.BatchNorm2d(10)
        self.conv3 = nn.Conv2d(10,1,32,1,0)

    
    def forward(self,imgs,labels):
        vec_1 = self.embed(torch.LongTensor(labels).cuda())
        vec_1 = vec_1.view(-1,5,32,32)
        vec_1 = F.leaky_relu(vec_1,0.2)
        out_1 = F.leaky_relu(self.BN1(self.conv1(imgs)),0.2)
        out_2 = F.leaky_relu(self.BN2(self.conv2(out_1)),0.2)
        #vec_2 = torch.cat((out_2,vec_1),1)
        vec_3 = torch.squeeze(self.conv3(out_2))
        return vec_3
        
        
class ResBlock_3(nn.Module):
    def __init__(self):
        super(ResBlock_3,self).__init__()
        self.seq = nn.Sequential(
            nn.Linear(2,2)
        )
        self.l = nn.Linear(1,2)
    def forward(self):
        pass




























