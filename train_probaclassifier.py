import torch as t
from torch.utils.data import DataLoader,random_split
from torch import optim
import pandas as pd
import numpy as np
from assets.fineclassifyData import fineClassifiyData
from assets.fineclassifier import fineClassifier
from torch import nn
from numpy import argmax

csv_name = 'test_res.csv'
csv_out_dir = '/home/antoine/antoine/cervical_model/pytorch_model/classification/TissueNet/baseline/'+csv_name
csv_out  =  pd.read_csv(csv_out_dir)
gts = np.array(csv_out['gt']).tolist()
probs = [[csv_out['0'][i],csv_out['1'][i],csv_out['2'][i],csv_out['3'][i]] for i in range(len(gts))]
data = fineClassifiyData(probs = probs,gts = gts)
device = t.device('cuda' if t.cuda.is_available() else 'cpu')

data_size = len(data)
n_train = int(data_size * 0.8)
train,val = random_split(data,[n_train,data_size - n_train])
train_loader = DataLoader(train,batch_size=16,shuffle=True)#使用DataLoader加载数据
val_loader = DataLoader(val,batch_size=16,shuffle=True)

net = fineClassifier()
net.to(device=device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(),lr = 1e-3,weight_decay=1e-5)
scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=20,gamma=0.1)
epochs = 60
def train():
    net.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for i_batch,batch_data in enumerate(train_loader):
            probs = batch_data['probs'].to(device = device)
            label = batch_data['gts'].to(device = device)
            pred = net(probs)
            pred = pred.squeeze()
            loss = criterion(pred,label)
            epoch_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step(epoch_loss)
                
        print('training loss is:' ,epoch_loss / n_train)

def test():
        net.eval()
        epoch_loss = 0
        gt = []
        pre = []
        for i_batch,batch_data in enumerate(val_loader):
            probs = batch_data['probs'].to(device = device)
            #print(vin.shape)
            label = batch_data['gts'].cpu()
            gt += label.numpy().tolist()

            pred = net(probs)
            pred = pred.squeeze().detach().cpu()

            loss = criterion(pred,label)
            epoch_loss += loss.item()

            #print(pred.shape)
            if len(list(pred.size())) == 1:
                pre.append(int(argmax(pred,0)))
            else:
                pre += argmax(pred,1).tolist()
        ac = float(sum(np.array(gt) == np.array(pre)))/float(len(gt))
        print('accuracy on test set is: ', ac)

        return gt,pre,epoch_loss
train()

gt,pre,_ = test()
print('gt',gt)
print('pre',pre)

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

print('score:',1 - scoring(gt,pre))

if 1-scoring(gt,pre) > 0.82:
    t.save(net.state_dict(),'assets/fineClassifier.pth')