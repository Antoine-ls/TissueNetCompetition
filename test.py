from numpy.core.fromnumeric import argmax
import torch as t
from torch.utils.data import DataLoader
from assets.TissueTestData import TissueTestAnno
from torchvision import transforms as T
from tqdm import tqdm
import numpy as np 
import torchvision as tv
import os 
from assets.ROI import ROI
import pandas as pd

from assets.img_calculs import seg_hsv_threshold,get_boxes_contours,get_polygons_contours
from assets.img_calculs import denoise_bilatera, denoise_erode
from assets.img2np import np_from_tif

transform = T.Compose([
    T.Resize(512),
    T.ToTensor()
])
csv_dir = '/mnt/hd1/antoine/TissueNet/TissueNet_Detect_Lesions_in_Cervical_Biopsies_-_Train_Labels.csv'
csv = pd.read_csv(csv_dir)
gt = []

resnet50 = tv.models.resnet18(pretrained = True)
resnet50.fc.out_features = 4

resnet50.load_state_dict(t.load('/home/antoine/antoine/cervical_model/pytorch_model/classification/TissueNet/epoch_final.pth'))

device = t.device('cuda' if t.cuda.is_available() else 'cpu')
resnet50.to(device=device)


def test():
    resnet50.eval()
    pre = []
    with tqdm(total = data_size, desc = f'evaluating',unit = 'img') as pbar:
        for i_batch,batch_data in enumerate(val_loader):
            img = batch_data['img'].to(device = device)
            pred = resnet50(img)
            pred = pred.squeeze().detach().cpu()
            if len(list(pred.size())) == 1:
                pre.append(int(argmax(pred,0)))
            else:
                pre += argmax(pred,1).tolist()
            pbar.update(img.shape[0])
    return np.array(pre)

df = pd.DataFrame(columns=['filename','samples','class0','class1','class2','class3','gt'])
tifs = os.listdir('/mnt/hd1/antoine/TissueNet/data/Train_Metadata/tif')
with tqdm(total = len(tifs), desc = f'evaluating',unit = 'img') as pbar2:
    for tif_name in tifs:
        assert(tif_name.endswith('.tif'))
        tif_dir = '/mnt/hd1/antoine/TissueNet/data/Train_Metadata/tif/' + tif_name
        img = np_from_tif(tif_dir, 6)
        #print(img.shape)
        gt = []
        for i in range(4):
            gt.append(int(csv[csv['filename'] == tif_name][str(i)]))
        mask = seg_hsv_threshold(img, disp=False, denoise_f=(denoise_erode, denoise_bilatera))
        #print(mask.shape)
        boxes, contours, _ = get_boxes_contours(img, mask, disp=False)
        R = ROI()
        R.set_boxes(name=tif_name,  H=img.shape[0],W=img.shape[1], boxes=boxes, coordinate='lu')
        boxes =R.get_boxes_normalized(name=tif_name, coordinate='lu')
        data = TissueTestAnno(tif_dir, rois = boxes, mask = mask,
                        transform=transform)
        data_size = len(data)
        val_loader = DataLoader(data,batch_size=32,shuffle=False)
        pre = test()
        print('In %d sampled pathes, there are %d class0, %d class1, %d class2, %d class3' %(len(data) ,sum(pre==0) 
        ,sum(pre==1) ,sum(pre==2) ,sum(pre==3)))
        print("ground truth is:", gt)
        df = df.append([{'filename':tif_name,'samples':len(data),'class0':sum(pre==0),'class1':sum(pre==0),'class2':sum(pre==0),'class3':sum(pre==0),'gt':gt}])
        pbar2.update(1)
    df.to_csv('pred.csv')
#score =  scoring(gt,pre)
#print('score is: ', 1 - score)