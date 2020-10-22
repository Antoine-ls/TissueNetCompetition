from torch import nn
import torchvision
from torchvision import transforms as T


class fineClassifier_v3(nn.Module):
    def __init__(self, num_out_features=4) -> None:
        super(fineClassifier_v3, self).__init__()
        self.resnet101 = torchvision.models.resnet101(pretrained = True)
        self.Linear0 = nn.Linear(1000, 512)
        self.Activ0 = nn.ReLU()
        self.Linear1 = nn.Linear(512, 64)
        self.Activ1 = nn.ReLU()
        self.Linear2 = nn.Linear(64, num_out_features)

    def forward(self, x):
        """
        input: x [512, 512]
        """
        out = self.resnet101(x)
        out = self.Linear0(out)
        out = self.Activ0(out)
        out = self.Linear1(out)
        out = self.Activ1(out)
        out = self.Linear2(out)
        return out

