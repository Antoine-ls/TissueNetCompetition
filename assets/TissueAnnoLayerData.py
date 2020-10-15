from torch.utils.data import DataLoader,Dataset
from PIL import Image
import matplotlib.pyplot as plt
import os
import torch
from torchvision import transforms as T
import numpy as np
import pandas as pd
import pyvips

class TissueAnnoLayerData(Dataset): #继承Dataset
    """
        polygon based dataset for training
    """
    def __init__(self, csv_dir, img_dir, page ,size=2048,transform=None): #__init__是初始化该类的一些基础参数
        self.csv_dir = csv_dir   #文件目录
        self.img_dir = img_dir   #文件目录
        self.transform = transform #变换
        self.page = page
        self.size = size
        self.csv = pd.read_csv(csv_dir)
        self.images = [x for x in self.csv['annotation_id']]
        self.tif_name = [x for x in self.csv['filename']]
        polygon_list=self.csv['geometry'].str.split(',')

        for polygon in polygon_list:
        # polygon list is a list of x y coordinates
            polygon[0]=polygon[0][9:]
            polygon[-1]=polygon[-1][:-2]
            for i in range(len(polygon)):
                polygon[i]=polygon[i][1:]
                # delete white space
                polygon[i]=[float(polygon[i].split(' ')[0]),float(polygon[i].split(' ')[1])]
        #         print(coord)
        self.polygons=np.ndarray(shape=(len(self.tif_name),4,2))  
        for i in range(5926):
            for j in range(4):
                for k in range(2):
                    self.polygons[i][j][k]=int(polygon_list[i][j][k])
        self.labels = np.array([x for x in self.csv['annotation_class']])
    
    def __len__(self):#返回整个数据集的大小
        return len(self.images)
    
    def __getitem__(self,index):#根据索引index返回dataset[index]
        image_index = self.tif_name[index]#根据索引index获取该图片
        img_path = os.path.join(self.img_dir, image_index)#获取索引为index的图片的路径名
        box=self.get_bbox_coord(self.polygons[index])
        tif_img=pyvips.Image.new_from_file(img_path,page=self.page)
        box = [int(x/2**(self.page-0)) for x in box]
        bbox = [max(1,int(box[0]-(self.size-box[2])*np.random.rand())),
            max(1,int(tif_img.height-box[1]-box[3]-(self.size-box[2])*np.random.rand())),
            min(tif_img.width - box[0] - 1,self.size),
            min(box[1] - 1,self.size)]
        #print(tif_img.width,tif_img.height)
        #print(bbox)
        try:
            area=pyvips.Region.new(tif_img).fetch(
            bbox[0],
            bbox[1],
            bbox[2],
            bbox[3]
            )
        except pyvips.error.Error:
            print(tif_img)
            print(tif_img.width,tif_img.height)
            print(bbox)
            exit()
        image=np.ndarray(buffer=area,
                    dtype=np.uint8,
                    shape=(bbox[2],bbox[3],3))
        image = Image.fromarray(image)
        label = self.labels[index]# 根据该图片的路径名获取该图片的label，具体根据路径名进行分割。
        sample = {'img':image,'label':label}#根据图片和标签创建字典
        
        if self.transform:
            sample['img'] = self.transform(sample['img'])#对样本进行变换
        return sample #返回该样本

    def get_bbox_coord(self,polygon):    
        xmin=np.amin(polygon,0)[0]
        xmax=np.amax(polygon,0)[0]
        ymin=np.amin(polygon,0)[1]
        ymax=np.amax(polygon,0)[1]
        width=xmax-xmin
        height=ymax-ymin
    #     print([int(xmin),int(ymin),int(width),int(height)])
        return [int(xmin),int(ymin),int(width),int(height)]