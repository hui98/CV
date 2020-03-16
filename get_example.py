import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import UT_PIL_Torchsave as SBLoader
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.optim as optim
import time
from torchvision import datasets, transforms
import models

class PILencode(object):
    def __call__(self,tensor):
        tmp = (tensor-0.5)/0.5
        return tmp
class PILdecode(object):
    def __call__(self,tensor):
        tmp = (tensor*0.5)+0.5
        return tmp

G = torch.load('/home/aistudio/work/github/CV/CV-master/G.pth')

D = torch.load('/home/aistudio/work/github/CV/CV-master/D.pth')
de = PILdecode()
eee = transforms.ToPILImage()

for i in range(0,5):
    for j in range(0,5):
        z_g = torch.randn(1,100).cuda()
        G.eval()
        #G.train()
        wishimg = G(z_g,[1])
        plt.subplot(5,5,i*5+j+1)
        bbb = eee(de(wishimg.cpu())[0])
        bbb.save('/home/aistudio/work/github/CV/CV-master/pics/'+str(i)+str(j)+'.png')
        plt.imshow(bbb)
nowtime = time.localtime(time.time())
plt.savefig('/home/aistudio/work/github/CV/CV-master/output/'+str(nowtime[3])+':'+str(nowtime[4])+':'+str(nowtime[5])+'.png')