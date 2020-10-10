from torch.utils.data import Dataset
import torch as t

class fineClassifiyData(Dataset): #继承Dataset
    """
        polygon based dataset for training
    """
    def __init__(self, probs , gts): #__init__是初始化该类的一些基础参数
        self.gts = gts   #文件目录
        self.probs = probs   #文件目录
    
    def __len__(self):#返回整个数据集的大小
        return len(self.gts)
    
    def __getitem__(self,index):#根据索引index返回dataset[index]
        probs = self.probs[index]
        gt = self.gts[index]
        sample = {'probs':t.Tensor(probs),'gts':int(gt)}
        return sample #返回该样本