from numpy.core.fromnumeric import argmax, var
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

from assets.img_calculs import seg_hsv_threshold,get_boxes_contours
from assets.img_calculs import denoise_bilatera, denoise_erode
from assets.img2np import np_from_tif

def scoring(gts,pres):
    score = 0.0
    score_matrix = [[0.0,0.1,0.7,1.0],
        [0.1,0.0,0.3,0.7],
        [0.7,0.3,0.0,0.3],
        [1.0,0.7,0.3,0.0]
    ]
    for i in range(len(gts)):
        score += score_matrix[gts[i]][pres[i]]
    score = score / len(gts)

    return score

transform = T.Compose([
    T.Resize(512),
    T.ToTensor()
])
os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'

# read csv and load tif filenames
csv_dir = '/mnt/hd1/antoine/TissueNet/TissueNet_Detect_Lesions_in_Cervical_Biopsies_-_Train_Labels.csv'
csv = pd.read_csv(csv_dir)
out_file_name = 'test_res_th_0.1.csv'
# output csv 
csv_out_dir = '/home/antoine/antoine/cervical_model/pytorch_model/classification/TissueNet/baseline/' + out_file_name
if not os.path.exists(csv_out_dir):
    temp = pd.DataFrame(columns=['filename','samples','0','1','2','3','gt','pre'])
    temp.to_csv(out_file_name,index = False)
csv_out  =  pd.read_csv(csv_out_dir)
processed_tifs = np.array(csv_out['filename']).tolist()

suppression_threshhold = 0.1

gt = []
tifs = csv['filename']
tifs_file = '/mnt/hd1/antoine/TissueNet/data/Train_Metadata/tif'
# load model
resnet50 = tv.models.resnet50(pretrained = False)
resnet50.fc.out_features = 4
resnet50.load_state_dict(t.load('/home/antoine/antoine/cervical_model/pytorch_model/classification/TissueNet/baseline/assets/epoch_final_res50.pth'))
device = t.device('cuda' if t.cuda.is_available() else 'cpu')
resnet50.to(device=device)


# initialize dataframe to store output information
df = pd.DataFrame(columns=['filename','samples','0','1','2','3','gt','pre'])

df = pd.concat([df,csv_out],axis = 0)

# test on one tif image
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

with tqdm(total = len(tifs), desc = f'evaluating',unit = 'img') as pbar2:
    pres = []
    gts = []
    for tif_name in tifs:
        if tif_name in processed_tifs:
            pbar2.update(1)
            continue
        tif_dir = os.path.join(tifs_file, tif_name)
        tif_label = np.argmax(csv[csv['filename'] == tif_name][['0','1','2','3']].values)
        gts.append(tif_label)
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
            df = df.append([{'filename':tif_name,'samples':0, '0':0,'1':0,'2':0,'3':0,'pre':0,'gt':tif_label}])
            pres.append(0)
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
            df = df.append([{'filename':tif_name,'samples':0,'0':0,'1':0,'2':0,'3':0,'pre':0,'gt':tif_label}])
            pres.append(0)
            pbar2.update(1)
            continue

        # initialize dataloader
        val_loader = DataLoader(data,batch_size=24,shuffle=False)
        pre = test()
        # naive voting principle is to filter class less than 5% and then choose highest
        result = np.array([sum(pre==0)/float(data_size),sum(pre==1)/float(data_size),sum(pre==2)/float(data_size),sum(pre==3)/float(data_size)])
        res = result> suppression_threshhold
        if res[3] == True:
            df = df.append([{'filename':tif_name,'samples':data_size,'0':result[0],'1':result[1],'2':result[2],'3':result[3],'pre':3,'gt':tif_label}])
            pres.append(3)
        elif res[2] == True:
            df = df.append([{'filename':tif_name,'samples':data_size,'0':result[0],'1':result[1],'2':result[2],'3':result[3],'pre':2,'gt':tif_label}])
            pres.append(2)
        elif res[1] == True:
            df = df.append([{'filename':tif_name,'samples':data_size,'0':result[0],'1':result[1],'2':result[2],'3':result[3],'pre':1,'gt':tif_label}])
            pres.append(1)
        else:
            df = df.append([{'filename':tif_name,'samples':data_size,'0':result[0],'1':result[1],'2':result[2],'3':result[3],'pre':0,'gt':tif_label}])
            pres.append(0)
        pbar2.update(1)
        print('predictions: ',result)
        print('label is:',tif_label)
        print('current score is: ',1 - scoring(gts,pres))

    # generate csv file from predictions
        df.to_csv(out_file_name,index = False)