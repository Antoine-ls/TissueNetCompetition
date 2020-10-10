from torch import nn

class fineClassifier(nn.Module):
    def __init__(self) -> None:
        super(fineClassifier,self).__init__()
        self.clas = nn.Sequential(
            nn.Linear(4,20),
            nn.ReLU(),
            nn.Linear(20,20),
            nn.ReLU(),
            nn.Linear(20,4)
        )

    def forward(self,x):
        return self.clas(x)
