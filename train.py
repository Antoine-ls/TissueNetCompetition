from os import write
from numpy.core.fromnumeric import argmax
from torch.optim import optimizer
import torch as t
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader,random_split
import torch.nn.functional as F
from torch import optim
from torchvision.transforms.transforms import Pad
from assets.TissueAnnoData import TissueDataAnno
from torchvision import transforms as T
from tqdm import tqdm
import numpy as np 
import torchvision as tv
from tensorboardX import SummaryWriter
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'

transform = T.Compose([
    T.RandomHorizontalFlip(0.5),
    T.RandomVerticalFlip(0.5),
    T.RandomRotation(30),
    T.Resize(512),
    T.ColorJitter(brightness=0.3,contrast=0.2),
    T.ToTensor()
])

data = TissueDataAnno('/home/antoine/antoine/cervical_model/pytorch_model/classification/TissueNet/TissueNet_Detect_Lesions_in_Cervical_Biopsies_-_Train_Annotations.csv',
                    '/home/antoine/antoine/cervical_model/pytorch_model/classification/TissueNet/data/Train_Annotations/jpg',
                    transform=transform)

#print(len(data))
data_size = len(data)
n_train = int(data_size * 0.8)
train,val = random_split(data,[n_train,data_size - n_train])
train_loader = DataLoader(train,batch_size=16,shuffle=True)#使用DataLoader加载数据
val_loader = DataLoader(val,batch_size=16,shuffle=True)

resnet50 = tv.models.resnet50(pretrained = True)
resnet50.fc.out_features = 4

#resnet50.load_state_dict(t.load('epoch_final.pth'))

device = t.device('cuda' if t.cuda.is_available() else 'cpu')
resnet50.to(device=device)

epochs = 50
criterion = nn.CrossEntropyLoss()
#optimizer = optim.RMSprop(resnet50.parameters(),lr = 1e-5, weight_decay=1e-8, momentum=0.9)
optimizer = optim.Adam(resnet50.parameters(),lr= 1e-4, weight_decay= 1e-6)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode = 'min', factor = 0.5, patience = 3)

writer = SummaryWriter(log_dir = 'train_logs')
sample = t.rand(1,3,512,512).to(device = device)
writer.add_graph(resnet50,input_to_model = (sample,))

def test():
        resnet50.eval()
        epoch_loss = 0
        gt = []
        pre = []
        with tqdm(total = data_size - n_train, desc = f'evaluating',unit = 'img') as pbar:
            for i_batch,batch_data in enumerate(val_loader):
                img = batch_data['img'].to(device = device)
                #print(vin.shape)
                label = batch_data['label'].cpu()
                gt += label.numpy().tolist()

                pred = resnet50(img)
                pred = pred.squeeze().detach().cpu()

                loss = criterion(pred,label)
                epoch_loss += loss.item()

                #print(pred.shape)
                if len(list(pred.size())) == 1:
                    pre.append(int(argmax(pred,0)))
                else:
                    pre += argmax(pred,1).tolist()

                pbar.update(img.shape[0])
        #print(np.array(gt))
        #print(np.array(pre))
        ac = float(sum(np.array(gt) == np.array(pre)))/float(len(gt))
        print('accuracy on test set is: ', ac)

        return gt,pre,epoch_loss


def train():
    for epoch in range(epochs):
        resnet50.train()
        epoch_loss = 0
        with tqdm(total = n_train, desc = f'Epoch {epoch+1}/{epochs}',unit = 'img') as pbar:
            for i_batch,batch_data in enumerate(train_loader):
                img = batch_data['img'].to(device = device)
                #print(vin.shape)
                label = batch_data['label'].to(device = device)

                pred = resnet50(img)
                pred = pred.squeeze()
            #    print(pred.shape)
            #   print(label.shape)

                loss = criterion(pred,label)
                epoch_loss += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                pbar.update(img.shape[0])
            
        print('training loss is:' ,epoch_loss / n_train)
        _,__,val_loss = test()
        scheduler.step(val_loss)

        writer.add_scalar('train_loss', epoch_loss, epoch)
        writer.add_scalar('val_loss',val_loss,epoch)

train()
t.save(resnet50.state_dict(),'epoch_final_layer0_aug_res50.pth')

#mmd.load_state_dict(t.load('epoch_100.pth'))
gt,pre,_ = test()
#print(gt)
#print(pre)

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

score =  scoring(gt,pre)
print('score is: ', 1 - score)