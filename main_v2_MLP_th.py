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
import pyvips
from assets.fineclassifier import fineClassifier

from assets.img_calculs import seg_hsv_threshold,get_boxes_contours
from assets.img_calculs import denoise_bilatera, denoise_erode
from assets.img2np import np_from_tif

transform = T.Compose([
    T.Resize(512),
    T.ToTensor()
])

# read csv and load tif filenames
csv_dir = 'data/test_metadata.csv'
csv = pd.read_csv(csv_dir)
tifs = csv['filename']

# load model
resnet50 = tv.models.resnet50(pretrained = False)
resnet50.fc.out_features = 4
resnet50.load_state_dict(t.load('assets/epoch_final_res50.pth'))
device = t.device('cuda' if t.cuda.is_available() else 'cpu')
resnet50.to(device=device)


# initialize dataframe to store output information
df = pd.DataFrame(columns=['filename','0','1','2','3'])

# test on one tif image
def test():
    resnet50.eval()
    pre = []
    with torch.no_grad(): # 会快一些
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

with tqdm(total = len(tifs), desc = f'evaluating',unit = 'img') as pbar2:
    proba_results = []
    for tif_name in tifs:
        tif_dir = 'data/' + tif_name
        layer = 8 # the layer for segmentation
        img = None
        while(img is None):
            try:
                img = np_from_tif(tif_dir, layer)
            except pyvips.error.Error:
                layer -= 1
            else:
                if img.shape[0] < 500 or img.shape[1] < 500: # too small segmentation layer means bad resolution
                    img = None
                    layer -= 1
        print('image for segmentation with size: ',img.shape)
        mask = seg_hsv_threshold(img, disp=False, denoise_f=(denoise_erode, denoise_bilatera))
        boxes, contours, _ = get_boxes_contours(img, mask, disp=False)
        if len(boxes) == 0:
            proba_results += [[1.0,0,0,0]]
            pbar2.update(1)
            continue
        R = ROI()
        R.set_boxes(name=tif_name,  H=img.shape[0],W=img.shape[1], boxes=boxes, coordinate='lu')
        boxes =R.get_boxes_normalized(name=tif_name, coordinate='lu')
        p = 0
        ind = 0
        data_size = 0

        # if there are too many patches (here more than 2000), downsample the tif image
        while ind == 0:
            # build dataset from tif and detected ROI
            if p == 0:
                data = TissueTestAnno(tif_dir, rois = boxes, mask = mask, stride = 2048, grid_size = 2048, page = p ,mask_page = layer,
                        transform=transform)
            elif p == 1:
                data = TissueTestAnno(tif_dir, rois = boxes, mask = mask, stride = 1536, grid_size = 1536, page = p ,mask_page = layer,
                        transform=transform)
            elif p == 2:
                data = TissueTestAnno(tif_dir, rois = boxes, mask = mask, stride = 1024, grid_size = 1024, page = p ,mask_page = layer,
                        transform=transform)
                ind = 1
            data_size = len(data)
            if(data_size < 500):
                ind = 1
            p += 1
        

        if data_size == 0:
            proba_results += [[1.0,0,0,0]]
            pbar2.update(1)
            continue

        # initialize dataloader
        val_loader = DataLoader(data,batch_size=12,shuffle=False)
        pre = test()

        # naive voting principle is to filter class less than 5% and then choose highest
        res = np.array([sum(pre==0)/float(data_size),sum(pre==1)/float(data_size),sum(pre==2)/float(data_size),sum(pre==3)/float(data_size)])
        proba_results += [res.tolist()]

        pbar2.update(1)

net = fineClassifier()
net.load_state_dict(t.load('assets/fineClassifier.pth'))
device = t.device('cuda' if t.cuda.is_available() else 'cpu')
net.to(device = device)

def test():
        net.eval()
        pre = []
        with torch.no_grad():
            probs = t.Tensor(proba_results).to(device = device)
            print(probs.shape)
            pred = net(probs)
            pred = pred.squeeze().detach().cpu()
            pre += argmax(pred,1).tolist()
        return pre
pre = test()

for i in range(len(pre)):
        if pre[i] == 3:
            df = df.append([{'filename':tifs[i],'0':0,'1':0,'2':0,'3':1}])
        elif pre[i] == 2:
            df = df.append([{'filename':tifs[i],'0':0,'1':0,'2':1,'3':0}])
        elif pre[i] == 1:
            df = df.append([{'filename':tifs[i],'0':0,'1':1,'2':0,'3':0}])
        else:
            df = df.append([{'filename':tifs[i],'0':1,'1':0,'2':0,'3':0}])
# generate csv file from predictions
df.to_csv('submission.csv',index = False)