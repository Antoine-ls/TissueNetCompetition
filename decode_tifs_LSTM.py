"""
这个脚本读取TIF文件，提取有细胞的区域，分割，收集每一个patch的预测值，记录到一个list里。
Example：
proba_results
[array([2, 2, 0, 3, 0... 0, 0, 3]), array([0, 1, 0, 0, 0... 0, 0, 0])]
"""
from numpy.core.fromnumeric import argmax
import torch as torch
from torch.utils.data import DataLoader
from assets.TissueTestDataLSTM import TissueTestAnno
from torchvision import transforms as T
from tqdm import tqdm
import numpy as np 
import torchvision
import os 
import pickle
from assets.ROI import ROI
import pandas as pd
import pyvips

from assets.img_calculs import seg_hsv_threshold,get_boxes_contours
from assets.img_calculs import denoise_bilatera, denoise_erode
from assets.img2np import np_from_tif


transform = T.Compose([
    T.Resize(512),
    T.ToTensor()
])

# test on one tif image
def predict_on_patches(val_loader):
    pre = []
    global model
    global DEVICE
    
    # with tqdm(total = len(val_loader), desc = f'evaluating',unit = 'img') as pbar:
    for i_batch, batch_data in enumerate(val_loader):
        img = batch_data['img'].to(device = DEVICE)
        pred = model(img)
        pred = pred.squeeze().detach().cpu()
        if len(list(pred.size())) == 1:
            pre.append(int(argmax(pred,0)))
        else:
            pre += argmax(pred,1).tolist()
            # pbar.update(img.shape[0])
    return np.array(pre)

def get_segmented_patches(tif_path):

    # choose propriate layer
    layer = 8 # the layer for segmentation
    img = None
    while(img is None and layer > 0):
        try:
            img = np_from_tif(tif_path, layer)
        except pyvips.error.Error:
            layer -= 1
        else:
            if img.shape[0] < 500 or img.shape[1] < 500: # too small segmentation layer means bad resolution
                img = None
                layer -= 1

    print('image for segmentation with size: ',img.shape)

    # get boxes
    mask = seg_hsv_threshold(img, disp=False, denoise_f=(denoise_erode, denoise_bilatera))
    boxes, contours, _ = get_boxes_contours(img, mask, disp=False)
    if len(boxes) == 0:
        return None, None, None, None
    R = ROI()
    R.set_boxes(name=tif_name, H=img.shape[0],W=img.shape[1], boxes=boxes, coordinate='lu')
    boxes_normalized = R.get_boxes_normalized(name=tif_name, coordinate='lu')
    
    p = 0
    ind = 0
    num_patches = 0

    # if there are too many patches (here more than 2000), downsample the tif image
    while ind == 0:
        # build dataset from tif and detected ROI
        if p == 0:
            patches = TissueTestAnno(tif_path, rois = boxes_normalized, mask = mask, stride = 2048, grid_size = 2048, page = p ,mask_page = layer,
                    transform=transform)
        elif p == 1:
            patches = TissueTestAnno(tif_path, rois = boxes_normalized, mask = mask, stride = 1536, grid_size = 1536, page = p ,mask_page = layer,
                    transform=transform)
        elif p == 2:
            patches = TissueTestAnno(tif_path, rois = boxes_normalized, mask = mask, stride = 1024, grid_size = 1024, page = p ,mask_page = layer,
                    transform=transform)
            ind = 1
        num_patches = len(patches)
        if(num_patches < 500):
            ind = 1
        p += 1
    

    if num_patches == 0:
        return None, None, None, None

    # initialize dataloader
    val_loader = DataLoader(patches,batch_size=12,shuffle=False)
    anchors_normalized = patches.get_anchors_normalized()
    anchor_group_info = patches.get_anchor_group()

    return val_loader, boxes_normalized, anchors_normalized, anchor_group_info

if __name__ == '__main__':

    # CSV_PATH = './data/test_metadata.csv'
    CSV_PATH = './data/work_metadata.csv'
    RESNET_MODEL_PATH = './assets/epoch_final_res50.pth'
    CLASSIFIER_MODEL_PATH = './assets/fineClassifier.pth'
    # DATA_DIR = './data/'
    DATA_DIR = '/mnt/hd1/antoine/TissueNet/data/Train_Metadata/tif/'
    OUTPUT_DIR = './train_data/proba_results2/'
    ANNOTAION_CSV_PATH = './train_data/TissueNet_Detect_Lesions_in_Cervical_Biopsies_-_Train_Labels.csv'

    feature_map_size = 64
    resolution  = 1 / feature_map_size # resolution of feature map

    MODEL_OUT_FEATURES = 4
    if torch.cuda.is_available():
        DEVICE = torch.device('cuda',index=2)
    else:
        DEVICE = torch.device('cpu')

    refine = False

    # read csv and load tif filenames
    tif_names = pd.read_csv(CSV_PATH)['filename']
    num_tif = len(tif_names)

    # read annotations
    annotation_df = pd.read_csv(ANNOTAION_CSV_PATH)

    # load model
    model = torchvision.models.resnet50(pretrained = False)
    model.fc.out_features = MODEL_OUT_FEATURES
    model.load_state_dict(torch.load(RESNET_MODEL_PATH))
    model.to(device=DEVICE)
    model.eval()


    with tqdm(total = len(tif_names), desc = f'evaluating',unit = 'img') as pbar2:
        try:
            for idx, tif_name in enumerate(tif_names):
                proba_results = dict()
                tif_path = os.path.join(DATA_DIR, tif_name)
                val_loader, boxes_normalized, anchors_normalized, anchor_group_info = get_segmented_patches(tif_path)
                
                if val_loader is None or boxes_normalized is None:
                    proba_results[tif_name] = None
                    pbar2.update(1)
                    continue


                prediction = predict_on_patches(val_loader)
                
                feature_map = np.zeros(shape=(feature_map_size,feature_map_size))
                idx_x = (anchors_normalized[:, 0] // resolution).astype(np.int32)
                idx_y = (anchors_normalized[:, 1] // resolution).astype(np.int32)
                feature_map[ idx_x, idx_y] = prediction + 1 # 0 - empty 1 - normal 2 - middle 3 - bad 4 - nightmare

                # get label and class of the whol tif
                label = np.array(annotation_df[annotation_df['filename'] == tif_name][['0','1','2','3']])
                annotation_class = np.argmax(label)

                #dump pickle objects
                proba_results= {"tif_name": tif_name, "prediction":prediction, "boxes_normalized":boxes_normalized, "anchors_normalized":anchors_normalized, "anchor_group_info": anchor_group_info, "feature_map": feature_map, "label": label, "annotation_class": annotation_class}
                
                output_path = os.path.join(OUTPUT_DIR, tif_name + '.obj')

                with open(output_path, 'wb') as f:
                    pickle.dump(proba_results, f)
                    print('end')

                pbar2.update(1)
        except RuntimeError as e:
            print(e)
            exit()
        else:
            print("UNKNOWN_ERROR")
            pbar2.update(1)