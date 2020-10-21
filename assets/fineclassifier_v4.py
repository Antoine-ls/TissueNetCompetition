from torch import nn
import torchvision

# 没有效果
class fineClassifier_v4(nn.Module):
    def __init__(self, num_out_features=4) -> None:
        super(fineClassifier_v4, self).__init__()

        self.Conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(5,5)) # out: [16, 60, 60]
        self.Activ1 = nn.ReLU()
        self.Pool1 = nn.MaxPool2d(kernel_size=(2,2)) # out [16, 30, 30]
        
        self.Conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3)) # out: [32, 28, 28]
        self.Activ2 = nn.ReLU()
        self.Pool2 = nn.MaxPool2d(kernel_size=(2, 2)) # out: [32, 14, 14]
        
        self.Conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(5, 5)) # out: [64, 10, 10]
        self.Activ3 = nn.ReLU()
        self.Pool3 = nn.MaxPool2d(kernel_size=(2,2)) # out: [64, 5, 5]
        
        self.Linear0 = nn.Linear(64 * 5 * 5, 256)
        self.Linear1 = nn.Linear(256, 64)
        self.Linear2 = nn.Linear(64, num_out_features)

    def forward(self, x):
        """
        input: x:[batch, 3, 512, 512]
        """
        
        out = self.Conv1(x)
        out = self.Activ1(out)
        out = self.Pool1(out) 
        
        out = self.Conv2(out) 
        out = self.Activ2(out)
        out = self.Pool2(out) 

        out = self.Conv3(out)
        out = self.Activ3(out)
        out = self.Pool3(out)

        out = out.view(-1, 64 * 5 * 5)
        out = self.Linear0(out)
        out = self.Linear1(out)
        out = self.Linear2(out)

        return out
