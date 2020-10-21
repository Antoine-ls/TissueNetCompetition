from torch.utils.data import DataLoader,Dataset
from PIL import Image
from torchvision import transforms as T
import numpy as np
import pickle
import numpy as np
import torch

class TissueTestAnno(Dataset): #继承Dataset
    '''
    Dataset consisting of generating test patches for preprocessed box,mask.
    '''
    def __init__(self, obj_path_list): #__init__是初始化该类的一些基础参数
        '''
        params:
            obj_list
        '''
        self.obj_path_list = obj_path_list
        
    def __len__(self):#返回整个数据集的大小
        return len(self.obj_path_list)
    
    def __getitem__(self,index):#根据索引index返回dataset[index]
        obj_path = self.obj_path_list[index]
        # 加载dump的obj文件
        with open(obj_path, 'rb') as f:
            obj = pickle.load(f)

        f_map = np.expand_dims(obj['feature_map'], axis=0)
        f_map = np.repeat(f_map, 3, 0) / 4 # 归一化
        sample = {'f_map': f_map, "label": obj['annotation_class']} # feature map （64x64 float32） 和 annotation_class （int)
        return sample