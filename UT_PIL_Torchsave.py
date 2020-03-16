import torch
import PIL
from torchvision import transforms
from PIL import Image
import os
'''
f = 'dataset.pth'
im = Image.open('test2.jpg')
im_list = []
im_list.append(im)
torch.save(im_list,f)
'''
'''
b = torch.load(f)
b[0].show()
c = transforms.ToTensor()
print(c(b[0]).size())
'''

def read_and_save(datapath,savepath):
    if not os.path.exists(savepath):
        dataset = []
        labels = {'man':1,'woman':2}
        for root, dirs, pics in os.walk(datapath, topdown=True):
            if root == (datapath+'/man'):   
                for pic in pics:
                    im = Image.open(os.path.join(root,pic))
                    dataset.append([im,0])
            if root == (datapath+'/womaan'):
                for pic in pics:
                    im = Image.open(os.path.join(root,pic))
                    dataset.append([im,1])
            if root == (datapath+'/cropped'):
                for pic in pics:
                    try:
                        im = Image.open(os.path.join(root,pic))
                    except:
                        os.system('rm  '+os.path.join(root,pic))
                    dataset.append([im,2])
        torch.save(dataset,savepath)
        print('dataset has been updated')

def data_load(pthpath):
    if not os.path.exists(pthpath):
        print('no such pth file')
        return None
    return torch.load(pthpath)

