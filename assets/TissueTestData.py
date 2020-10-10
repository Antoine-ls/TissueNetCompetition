from torch.utils.data import DataLoader,Dataset
from PIL import Image
from torchvision import transforms as T
import numpy as np
import pyvips


def bbox_divide(box,stride, grid_size,mask,scale,height,width):
    '''
    Use detected box to generate anchor point for sampling and use mask to filter background area in boxes.
    params:
        box: detected roi boxes in preprocessing
        stride: stride of sliding window
        grid_size: size of sliding window
        mask: segmented mask in preprocessing
        scale: rate between image for preprocessing and image for classification
        height,width: height and width of image for classification
    '''
    if box[0]>min(box[2]+grid_size,width - grid_size-1) or box[1]>min(box[3]+grid_size,height-grid_size-1):
        return []
    x,y = np.mgrid[box[0]:min(box[2]+grid_size,width - grid_size-1):stride,box[1]:min(box[3]+grid_size,height-grid_size-1):stride]
    anchors = []
    points = np.c_[x.ravel(),y.ravel()]

    # filter anchors not in polygon
    anchors = [x for x in points if mask[int(x[1]/scale),int(x[0]/scale)]==255]
    return anchors


class TissueTestAnno(Dataset): #继承Dataset
    '''
    Dataset consisting of generating test patches for preprocessed box,mask.
    '''
    def __init__(self, img_dir, rois, mask, page = 0 ,mask_page = 6,transform=None,stride = 512, grid_size = 1024): #__init__是初始化该类的一些基础参数
        '''
        params:
            img_dir: dir of tif image
            rois: list of boxes
            mask: segmentation mask 
            page: page to do classification (when too many patches, increase page to save time)
            mask_page: page of preprocessing
        '''
        self.grid_size = grid_size
        self.scale = 2**(mask_page - page)
        self.mask = mask
        self.img = pyvips.Image.new_from_file(img_dir,page=page)
        self.height = self.img.height
        self.width = self.img.width
        print('tif image has shape', self.height, ' ', self.width)
        self.ROIs = [[int(x[0] * self.width), int(x[1] * self.height), int((x[2] + x[0]) * self.width), int((x[1]+x[3]) * self.height)] for x in rois]
        print('There are ', len(self.ROIs),' ROIs detected.')
        self.transform = transform #变换
        self.anchors = []
        anchors = []
        for bbox in self.ROIs:
            anchors += bbox_divide(bbox,stride,grid_size,self.mask,self.scale,self.height,self.width)
        # double check out of array error, may be redundant 
        for anchor in anchors:
            if anchor[0] + self.grid_size < self.width - 1 and anchor[1] + self.grid_size < self.height - 1:
                self.anchors.append(anchor)
        print('ROIs are divided into: ',len(self.anchors), ' patches for evaluating.')

    
    def __len__(self):#返回整个数据集的大小
        return len(self.anchors)
    
    def __getitem__(self,index):#根据索引index返回dataset[index]
        anchor = self.anchors[index]
        img = pyvips.Region.new(self.img).fetch(anchor[0],anchor[1],min(self.grid_size,self.width - anchor[0]-1),
                min(self.height - anchor[1]-1, self.grid_size))
        img = np.ndarray(buffer=img,
                    dtype=np.uint8,
                    shape=(self.grid_size,self.grid_size,3))
        img = Image.fromarray(img).convert("RGB")
        sample = {'img':img}
        
        if self.transform:
            sample['img'] = self.transform(sample['img'])#对样本进行变换
        return sample #返回该样本