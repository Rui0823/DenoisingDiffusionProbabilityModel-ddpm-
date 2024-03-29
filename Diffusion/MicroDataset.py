from torch.utils import data
import os
# from torchvision import transforms
from PIL import Image
class MicroDataSet(data.Dataset):
    def __init__(self, root,mode,transforms=None):
        self.data=[]
        if mode=="train":
            self.dataroot=os.path.join(root,"train")
            self.data=os.listdir(self.dataroot)
        elif mode=="test":
            self.dataroot=os.path.join(root,"test")
            self.data = os.listdir(self.dataroot)
        self.transforms=transforms

    def __getitem__(self, index):
        img=Image.open(os.path.join(self.dataroot,self.data[index]))
        if self.transforms!=None:
            img=self.transforms(img)
        return img

    def __len__(self):
        return len(self.data)
