import pandas as pd
import numpy as np
from assets.fineclassifier import fineClassifier
import torch as t
from numpy import argmax

def scoring(gts,pres):
    score = 0.0
    score_matrix = [[0.0,0.1,0.7,1.0],
        [0.1,0.0,0.3,0.7],
        [0.7,0.3,0.0,0.3],
        [1.0,0.7,0.3,0.0]
    ]
    conf_matrix = np.zeros((4,4))
    for i in range(len(gts)):
        score += score_matrix[gts[i]][pres[i]]
        conf_matrix[gts[i]][pres[i]] += 1
    score = score / len(gts)

    return score,conf_matrix
csv_name = 'test_res_th_0.1.csv'
csv_out_dir = '/home/antoine/antoine/cervical_model/pytorch_model/classification/TissueNet/baseline/'+csv_name
csv_out  =  pd.read_csv(csv_out_dir)
gts = np.array(csv_out['gt']).tolist()

# this is threshhold method
pres = np.array(csv_out['pre']).tolist()

# here is classifier method based on four probabilities.
class0 = np.array(csv_out['0']).tolist()
class1 = np.array(csv_out['1']).tolist()
class2 = np.array(csv_out['2']).tolist()
class3 = np.array(csv_out['3']).tolist()
classes = [[class0[i],class1[i],class2[i],class3[i]] for i in range(len(class0))]
net = fineClassifier()
net.load_state_dict(t.load('assets/fineClassifier.pth'))
device = t.device('cuda' if t.cuda.is_available() else 'cpu')
net.to(device = device)

def test():
        net.eval()
        gt = gts
        pre = []
        probs = t.Tensor(classes).to(device = device)

        pred = net(probs)
        pred = pred.squeeze().detach().cpu()
        if len(list(pred.size())) == 1:
            pre.append(int(argmax(pred,0)))
        else:
            pre += argmax(pred,1).tolist()
        ac = float(sum(np.array(gt) == np.array(pre)))/float(len(gt))
        print('accuracy on test set is: ', ac)
        return pre
pre = test()
score,conf = scoring(gts,pres)
score2,conf2 = scoring(gts,pre)
print('threshhold method:')
print('final score is:',1 - score)
print('confusion matrix is:')
print(conf)

print('perceptron method:')
print('final score is:',1 - score2)
print('confusion matrix is:')
print(conf2)