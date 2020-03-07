
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Input Block

        self.drop_val = 0.1
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3,3), padding=1, bias=False), #RF 3
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout2d(self.drop_val),
            nn.Conv2d(in_channels=32, out_channels=128, kernel_size=(3,3), padding=1, bias=False), # RF 5
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout2d(self.drop_val),
        )
        self.MP1 = nn.Sequential(
            nn.MaxPool2d(2,2),
            nn.Conv2d(in_channels=128, out_channels=32, kernel_size=(1,1), bias=False)  # RF 6
        )

        # Convolution Block 2
        self.convblock2=nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3,3), padding=1, bias=False), #RF 14
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(self.drop_val),
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=(3,3), padding=1, bias=False),   # RF 18
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout2d(self.drop_val),
        )
        self.MP2 = nn.Sequential(
            nn.MaxPool2d(2,2),  # RF 20
            nn.Conv2d(in_channels=256, out_channels=64, kernel_size=(1,1), bias=False)
        )

         # Convolution Block 3
        self.convblock3=nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3,3), padding=1,groups = 64, bias=False), #RF 28
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout2d(self.drop_val),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3,3), padding=2,dilation=2, bias=False),   # RF 36
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout2d(self.drop_val),
        )

        self.MP3 = nn.Sequential(
            nn.MaxPool2d(2,2),  # RF 40
            nn.Conv2d(in_channels=256, out_channels=32, kernel_size=(1,1), bias=False)
        )

         # Convolution Block 4
        self.convblock4=nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=256, kernel_size=(3,3),groups=32, padding=1, bias=False), #RF 56
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout2d(self.drop_val),
            nn.AvgPool2d(kernel_size=4),
            nn.Conv2d(in_channels=256, out_channels=10, kernel_size=(1,1),padding=0, bias=False)
        )

    def forward(self,x):
      x = self.convblock1(x)
      x = self.MP1(x)
      x = self.convblock2(x)
      x = self.MP2(x)
      x = self.convblock3(x)
      x = self.MP3(x)
      x = self.convblock4(x)
      x = x.view(-1,10)

      return F.log_softmax(x, dim=-1)


