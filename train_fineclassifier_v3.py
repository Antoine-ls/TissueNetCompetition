"""
这个脚本负责训练一个分类器

"""
from numpy.core.fromnumeric import argmax
import torch as torch
from torch.utils.data import DataLoader
from assets.TissueTestData import TissueTestAnno
from torchvision import transforms as T
from tqdm import tqdm
import numpy as np 
import torchvision
import os 
import pickle
from assets.ROI import ROI
import pandas as pd
import pyvips
from assets.fineclassifier_v3 import fineClassifier_v3 as fineClassifier
import glob
from assets.img_calculs import seg_hsv_threshold,get_boxes_contours
from assets.img_calculs import denoise_bilatera, denoise_erode
from assets.img2np import np_from_tif
from assets.TissueFeatureMapDataLSTM import TissueTestAnno
import torch.nn as nn
import datetime
import time
import uuid

if __name__ == '__main__':
    # The repository is located at /mnt/hd1/yutong/TissueNetCompetition
    CLASSIFIER_MODEL_PATH = './train_data/models/1b76ac96_50_1_0.5.pth'
    DATA_DIR = './data/'
    PROBA_RESULTS_DIR = './train_data/proba_results_tmp/'
    OUTPUT_DIR = './train_data/models/'
    refine = False
    

    if torch.cuda.is_available():
        DEVICE = torch.device('cuda', 1)
    else:
        DEVICE = torch.device('cpu')

    BATCH_SIZE = 64
    MODEL_OUT_FEATURES = 4
    NUM_EPOCH = 200
    LR = 2e-4

    # load model
    classifier_model = fineClassifier(num_out_features=MODEL_OUT_FEATURES)
    if refine: # refine the model
        classifier_model.load_state_dict(torch.load(CLASSIFIER_MODEL_PATH))
        OUTPUT_PATH = CLASSIFIER_MODEL_PATH
        print("Successfully load state_dict")

    classifier_model.to(device=DEVICE)
    classifier_model.train()

    # get obj path
    obj_path_list = glob.glob(os.path.join(PROBA_RESULTS_DIR, '*.obj'))
    obj_path_list_len = len(obj_path_list)

    # divide samples
    ratio = 0.7
    train_path_list = obj_path_list[0:int(obj_path_list_len * ratio)]
    test_path_list = obj_path_list[int(obj_path_list_len * ratio):]

    train_dataset = TissueTestAnno(train_path_list)
    test_dataset = TissueTestAnno(test_path_list)

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # define loss function, optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(classifier_model.parameters(), lr= LR, weight_decay= 1e-6)


    # Train
    classifier_model.train()
    with tqdm(total = NUM_EPOCH, desc = f'training',unit = 'batch') as pbar:
        for num_epoch in range(NUM_EPOCH):
            running_loss = 0
            for batch_idx, samples in enumerate(train_dataloader):
                f_map = samples["f_map"].to(DEVICE, dtype=torch.float32)
                label = samples["label"].to(DEVICE, dtype=torch.int32)

                optimizer.zero_grad()
                pred = classifier_model(f_map)
                loss = criterion(pred, label.long())
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                pbar.set_postfix({'loss' : '{0:1.5f}'.format(running_loss / (batch_idx + 1))})

            pbar.update(1)

    # Evaluation
    classifier_model.eval()
    torch.cuda.empty_cache()
    time.sleep(1)

    evaluating_loss = 0
    true_cnt = 0

    with torch.no_grad(): # save memory
        with tqdm(total = len(test_dataloader), desc = f'evaluating',unit = 'obj') as pbar:
            for batch_idx, samples in enumerate(test_dataloader):
                f_map = samples["f_map"].to(DEVICE, dtype=torch.float32)
                label = samples["label"].to(DEVICE, dtype=torch.int32)

                pred = classifier_model(f_map)
                loss = criterion(pred, label.long())
                true_cnt += sum(torch.argmax(pred, -1) == label.long()).item() # number of correctly classified items

                evaluating_loss += loss.item()
                pbar.set_postfix({'loss' : '{0:1.5f}'.format(evaluating_loss / (batch_idx + 1))})
                pbar.set_postfix({'accuracy' : '{0:1.5f}'.format(true_cnt / ((batch_idx + 1) * BATCH_SIZE))})

                pbar.update(1)

    avg_evaluating_loss = evaluating_loss / len(test_dataloader)
    accuracy = true_cnt / len(test_dataset)

    #save model
    if not refine:
        OUTPUT_PATH = os.path.join(OUTPUT_DIR,'v2_{}_{}_{:.5f}_{:.5f}.pth'.format(str(uuid.uuid1())[0:8],NUM_EPOCH, avg_evaluating_loss, accuracy))
    else:
        OUTPUT_PATH = CLASSIFIER_MODEL_PATH
    torch.save(classifier_model.state_dict(), OUTPUT_PATH)
    print("state dicts saved to {}".format(OUTPUT_PATH))
    print('end') 