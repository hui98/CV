from torchvision import transforms
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import Saberface_dataset as saber

trans =transforms.Compose([

        transforms.Resize((128,128)),

        transforms.ToTensor(),
        #transforms.Lambda(lambda x: x.repeat(3,1,1)),
        #transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))

])

c = transforms.Compose([
    transforms.Resize((32,32)),
    transforms.ToTensor(),
    #PILencode()
    ])
mydataset = saber.SFDataset(c)
loader = DataLoader(dataset=mydataset, batch_size=1)

train_loader = torch.utils.data.DataLoader(

	datasets.MNIST('data', train=True, download=True, transform=trans),

	batch_size=1, shuffle=True)

eee = transforms.ToPILImage()

#eee(de(wishimg.cpu())[0]).save(str(nowtime[3])+':'+str(nowtime[4])+':'+str(nowtime[5])+'.png')
d = saber.PILencode()
conv1 = nn.Conv2d(1,25,6,2,2)
en = saber.PILdecode()
for i,label in train_loader:
    
    b = conv1(i)
    print(b.size())
    '''
    eee(i[0]).save('test.png')
    b = d(i.view(4*32*32))
    ddd = en(b)
    for t in range(0,4*1024):
        print(b[t])
    for t in range(0,4*1024):
        print(ddd[t])
    print(label)
    '''
    break