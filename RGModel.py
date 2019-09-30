import torch
import torch.nn as nn
from torchvision.models import resnet50
import math
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

class RGModel(nn.Module):
    def __init__(self,embedding_size,num_classes,pretrained=False):
        super(RGModel, self).__init__()
        self.model = resnet50(pretrained)
        self.embedding_size = embedding_size
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.model.fc = nn.Linear(2048, num_classes)

    def forward(self, x):

        x = self.model.conv1(x)
        #print('conv1:',x.shape)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        #print('maxpool:',x.shape)
        x = self.model.layer1(x)
        #print('Layer1:',x.shape)
        x = self.model.layer2(x)
        #print('Layer2:',x.shape)
        x = self.model.layer3(x)
        #print('Layer3:',x.shape)
        x = self.model.layer4(x)
        #print('Layer4:',x.shape)
        x = self.model.avgpool(x)
        #print('avgpool',x.shape)
        x = x.view(x.size(0), -1)
        x = self.model.fc(x)

        return x




