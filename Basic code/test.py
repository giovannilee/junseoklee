from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as dset
from torch.autograd import Variable
from torch.autograd import Function
import torch.backends.cudnn as cudnn
import numpy as np
import os
import torchvision
from PIL import Image
import pickle
from inceptionv4 import InceptionV4
from collections import Counter
train_dir = 'train'
test_dir = 'test'

embedding_size = 256
class Scale(object):
    def __init__(self,size,interpolation=Image.BILINEAR):
        assert isinstance(size,int) or (isinstance(size, collections.lterable) and len(size) ==2)
        self.size = size
        self.interpolation = interpolation

    def __call__(self,img):
        if isinstance(self.size, int):
            w, h = img.size

            if w <= h:
                ow = self.size
                oh = int(self.size * h / w)
                return img.resize((ow,ow), self.interpolation)
            else:
                oh = self.size
                ow = int(self.size * w / h)
                return img.resize((oh,oh), self.interpolation)
        else:
            return img.resize(self.size, self.interpolation)

def find_name(cls_num):
    class_name = sorted(os.listdir(train_dir))
    return class_name[cls_num]

transform = transforms.Compose([
                         Scale(224),
                         transforms.ToTensor(),
                         transforms.Normalize(mean = [ 0.5, 0.5, 0.5 ],
                                               std = [ 0.5, 0.5, 0.5 ]) ])
test_dir = dset.ImageFolder('test',transform=transform)
test_loader = torch.utils.data.DataLoader(dataset=test_dir, batch_size = 1, shuffle=False)

def getlabel(x):
    with open("test/label/{}.pickle".format(x),"rb") as fr:
        label = pickle.load(fr)
    label = np.argmax(label)
    return label

name_list = os.listdir(train_dir)
model = InceptionV4( 10)

model.eval()
model.cuda()

for i in range(1,2):
    model.load_state_dict(torch.load('checkpoints/checkpoint_{}.pt'.format(i)))
    acc = 0
    tot = 0
    for batch_idx, data in enumerate(test_loader):
        img = data[0]
        label = data[1]
        true = label
        img = img.cuda()
        label = label.cuda()

        cls = model(img)
        cls = (cls.data).cpu().numpy()

        pred = np.argmax(cls)

        if pred == true:
            acc = acc + 1
        tot = tot + 1
        loading = 100 * (batch_idx / len(test_loader))
        print('loading... : {}%\r'.format(round(loading, 1)), end='')
    print('accuaracy : {}'.format(100*(acc/tot)))




