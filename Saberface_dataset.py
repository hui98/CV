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
class SFDataset(Dataset):
    def __init__(self,transform=None):
        super(SFDataset,self).__init__()
        SBLoader.read_and_save('./dataset','dataset.pth')
        self.PILimgs = SBLoader.data_load('dataset.pth')
        self.transform = transform
    def __getitem__(self, index):
        img_label = self.PILimgs[index]
        img = img_label[0]
        label = img_label[1]
        if self.transform is not None:
            img = self.transform(img)
        return img,label
    def __len__(self):
        return len(self.PILimgs)
        
def weight_init(m):
    # weight_initialization: important for wgan
    class_name=m.__class__.__name__
    if class_name.find('Conv')!=-1:
        m.weight.data.normal_(0,0.02)
    elif class_name.find('Norm')!=-1:
        m.weight.data.normal_(1.0,0.02)
#     else:print(class_name)

class PILencode(object):
    def __call__(self,tensor):
        tmp = (tensor-0.5)/0.5
        return tmp
class PILdecode(object):
    def __call__(self,tensor):
        tmp = (tensor*0.5)+0.5
        return tmp

    
if __name__ == '__main__':
    trans =transforms.Compose([

        transforms.Resize((128,128)),

        transforms.ToTensor(),
        PILencode()
    ]) 
    
    # init
    c = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor(),
    PILencode()])
    
    
    mydataset = SFDataset(c)
    
    
    print('是否加载现有模型？')
    
    is_load = input()
    if  is_load == '1':
        G = torch.load('G.pth')
        D = torch.load('D.pth')
    else:
        G = models.Generator()
        print('!!1')
        D = models.Discriminator()
        G.apply(weight_init)
        D.apply(weight_init)
    G.cuda()
    D.cuda()
    
    epoch = 50
    lr = 0.0001
    
    #!!!!!!!!!!!!!!!!
    batch = 256
    #one=torch.ones(1).cuda()
    #mone=-1*one
    #print(mone)
    loader = DataLoader(dataset=mydataset, batch_size=batch,shuffle=True)
    
    
    MNIST_loader = torch.utils.data.DataLoader(

	datasets.MNIST('data', train=True, download=True, transform=trans),

	batch_size=batch, shuffle=True)
	
	
	
	
    G_optimizer = optim.RMSprop(G.parameters(), lr=lr)

    D_optimizer = optim.RMSprop(D.parameters(), lr=lr)
    criterion = nn.BCELoss()
    de = PILdecode()
    eee = transforms.ToPILImage()
    j = 0
    k = 0
    d_loss = 0
    g_loss = 0
    save_num = 0
    time_num = 0
    D.train()
    G.train()
    wishimg = None
    for i in range(0,epoch):     
        time_num = 0
        for realimg,label in loader:
            # train d
            j+=1
            k+=1
            save_num+=1
            time_num+=1
            if k == 1:
                k = 0
                D.zero_grad()
                out1 = D(realimg.cuda(),label)
                #d_loss_real = criterion(out,torch.ones(batch).cuda())
                d_loss_real = torch.mean(out1)
                z_d = torch.randn(batch,100).cuda()
                fakeimg = G(z_d,label)
                
                out2 = D(fakeimg,label)
                d_loss_fake = torch.mean(out2)
                #d_loss_fake = criterion(out,torch.zeros(batch).cuda())
                #d_loss = out1+out2
                d_loss =  d_loss_fake - d_loss_real
                #d_loss = d_loss_real
                d_loss.backward()
                D_optimizer.step()
            for p in D.parameters():
                p.data.clamp_(-0.01,0.01)
            realimg.cpu()
            #train g
            if j == 5:
                j = 0
                G.zero_grad()
                D.zero_grad()
                z_g = torch.randn(batch,100).cuda()
                wishimg = G(z_g,label)
                out = D(wishimg,label)
                g_loss = -torch.mean(out)
                #g_loss = criterion(out,torch.ones(batch).cuda())
                g_loss.backward()
                G_optimizer.step()
                #print(str(i)+'/'+str(epoch)+'次,    D_loss:'+str(d_loss)+'  G_loss:'+str(g_loss))
                
            if save_num == 100:
                save_num = 0
                torch.save(G,'G.pth')
                torch.save(D,'D.pth')
                nowtime = time.localtime(time.time())
                z_g = torch.randn(1,100).cuda()
                G.eval()
                wishimg = G(z_g,[1])
                eee(de(wishimg.cpu())[0]).save(str(nowtime[3])+':'+str(nowtime[4])+':'+str(nowtime[5])+'.png')
                G.train()
                print('model save')
                print(str(i)+'/'+str(epoch)+'   第'+str(time_num)+'次,    D_loss:'+str(d_loss)+'  G_loss:'+str(g_loss))
    torch.save(G,'G.pth')
    torch.save(D,'D.pth')
    G.eval()
    z_g = torch.randn(1,100).cuda()
    wishimg = G(z_g,[1])   
    nowtime = time.localtime(time.time())
    eee(de(wishimg.cpu())[0]).save(str(nowtime[3])+':'+str(nowtime[4])+':'+str(nowtime[5])+'.png')

        
        
        
        
        
        
        
        
        
        
        