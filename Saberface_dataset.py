import torch
from torch.utils.data import Dataset, DataLoader
import UT_PIL_Torchsave as SBLoader
from torchvision import transforms
class SFDataset(Dataset):
    def __init__(self,transform=None):
        super(SFDataset,self).__init__()
        SBLoader.read_and_save('.\dataset','dataset.pth')
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
        
if __name__ == '__main__':
    c = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor(),
    ])
    d = transforms.ToPILImage()
    mydataset = SFDataset(c)
    loader = DataLoader(dataset=mydataset, batch_size=1)
    for img,label in loader:
        print(img.size())
        print(label)
        e = d(img[0])
        e.show()