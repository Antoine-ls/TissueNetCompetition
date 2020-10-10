from torch.utils.data import DataLoader,Dataset
from PIL import Image
import matplotlib.pyplot as plt
import os
import torch
from torchvision import transforms as T
import numpy as np
import pandas as pd

transform = T.Compose([
    T.ToTensor()
]   
)

class TissueDataAnno(Dataset): #继承Dataset
    """
        polygon based dataset for training
    """
    def __init__(self, csv_dir, img_dir, transform=None): #__init__是初始化该类的一些基础参数
        self.csv_dir = csv_dir   #文件目录
        self.img_dir = img_dir   #文件目录
        self.transform = transform #变换
        self.csv = pd.read_csv(csv_dir)
        self.images = [x for x in self.csv['annotation_id']]
        self.labels = np.array([x for x in self.csv['annotation_class']])
    
    def __len__(self):#返回整个数据集的大小
        return len(self.images)
    
    def __getitem__(self,index):#根据索引index返回dataset[index]
        image_index = self.images[index] + '.jpeg'#根据索引index获取该图片
        img_path = os.path.join(self.img_dir, image_index)#获取索引为index的图片的路径名
        img = Image.open(img_path)# 读取该图片
        label = self.labels[index]# 根据该图片的路径名获取该图片的label，具体根据路径名进行分割。
        sample = {'img':img,'label':label}#根据图片和标签创建字典
        
        if self.transform:
            sample['img'] = self.transform(sample['img'])#对样本进行变换
        return sample #返回该样本