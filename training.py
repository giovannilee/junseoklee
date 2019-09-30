from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as dset
import matplotlib.pyplot as plt

from torch.autograd import Variable
from torch.autograd import Function
import torch.backends.cudnn as cudnn
import os
import numpy as np
from PIL import Image
import collections
from torch import optim
from RGModel import RGModel
import pickle
from radam import RAdam
# Model options    
train_dir = 'Newdata/train/shiftblur'
epochs=300
embedding_size=256
batch_size=20

lr=0.1
lr_decay=1e-4
wd=0.0
opt='adagrad'
gpu_id='0'
seed=0

# set the device to use by setting CUDA_VISIBLE_DEVICES env variable in
# order to prevent any memory allocation on unused GPUs
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
np.random.seed(seed)

class Scale(object):
    def __init__(self, size, interpolation=Image.BILINEAR):
        assert isinstance(size, int) or (isinstance(size, collections.Iterable) and len(size) == 2)
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img):
        if isinstance(self.size, int):
            w, h = img.size
            if (w <= h and w == self.size) or (h <= w and h == self.size):
                return img
            if w < h:
                ow = self.size
                oh = int(self.size * h / w)
                return img.resize((ow, ow), self.interpolation)
            else:
                oh = self.size
                ow = int(self.size * w / h)
                return img.resize((oh, oh), self.interpolation)
        else:
            return img.resize(self.size, self.interpolation)

transform = transforms.Compose([
                         Scale(224),
                         transforms.ToTensor(),
                         #transforms.Normalize(mean = [ 0.5, 0.5, 0.5 ],
                           #                    std = [ 0.5, 0.5, 0.5 ])
                     ])
dir_length = len(os.listdir(train_dir))
loader_length = int((dir_length)/(batch_size))
#print('dl:',dir_length)
#print('ll:',loader_length)
def imcat(batch_idx, batch_size):
    im_storage = []
    start = (batch_idx)*(batch_size)
    for i in range(batch_size):
        im = Image.open("Newdata/train/shiftblur/{}.jpg".format(start+i))
        im = (transform(im)+1)*100
        im_storage.append(im)
    batched_im = np.stack(im_storage, axis=0) 
    return_value = torch.from_numpy(batched_im)

    return return_value

def getlabel(batch_idx, batch_size):
    label_storage = [] 
    start = (batch_idx)*(batch_size)
    for i in range(batch_size):
        with open("Newdata/train/label/{}.pickle".format(start+i),"rb") as fr:
            label = pickle.load(fr)
        #label = np.argmax(label)
        label_storage.append(label)
    batched_label = np.stack(label_storage, axis=0)
    return_value = torch.from_numpy(batched_label)

    return return_value

def main():
    # instantiate model and initialize weights
    #model = resnet.resnet50(pretrained=False)
    model = RGModel(embedding_size, 60, pretrained=True)
    model.cuda()
    #optimizer = create_optimizer(opt, model, lr)
    optimizer = RAdam(model.parameters())

    train_losses = []
    avg_train_losses = []
    valid_losses = []
    avg_valid_losses = []
    n_epochs = epochs
    print('train epochs : {}'.format(n_epochs))
    count = 0
    for epoch in range(1, n_epochs + 1):
    
        model.train()
        train_loss = 0
        for batch_idx in range(loader_length):
            img = imcat(batch_idx, batch_size)
            label = getlabel(batch_idx, batch_size)

            img = img.cuda()
            label = label.cuda()
            
            img = Variable(img).float()
            label = Variable(label).float()
            label_split = torch.split(label, 10, dim=1)

            cls = model(img)
            cls_b = torch.split(cls, 10, dim=1)

            criterion = nn.CrossEntropyLoss()

            loss1 = criterion(cls_b[0], torch.argmax(label_split[0],dim=1))
            loss2 = criterion(cls_b[1], torch.argmax(label_split[1],dim=1))
            loss3 = criterion(cls_b[2], torch.argmax(label_split[2],dim=1))
            loss4 = criterion(cls_b[3], torch.argmax(label_split[3],dim=1))
            loss5 = criterion(cls_b[4], torch.argmax(label_split[4],dim=1))
            loss6 = criterion(cls_b[5], torch.argmax(label_split[5],dim=1))

            loss = loss1 + loss2 + loss3 + loss4 + loss5 + loss6

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss = train_loss + loss.item()
            loading = 100*(batch_idx/loader_length)
            print('loading... : {}%\r'.format(round(loading,1)),end='')
        torch.save(model.state_dict(), 'checkpoint/checkpoint_{}.pt'.format(epoch))
        print('{} epochs done'.format(epoch))
        print('train_loss : ', train_loss)

def adjust_learning_rate(optimizer):
    """Updates the learning rate given the learning rate decay.
    The routine has been implemented according to the original Lua SGD optimizer
    """
    for group in optimizer.param_groups:
        if 'step' not in group:
            group['step'] = 0
        group['step'] += 1

        group['lr'] = lr / (1 + group['step'] * lr_decay)

def create_optimizer(opt, model, new_lr):
    # setup optimizer
    if opt == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=new_lr,
                              momentum=0.9, dampening=0.9,
                              weight_decay=wd)
    elif opt == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=new_lr,
                               weight_decay=wd)
    elif opt == 'adagrad':
        optimizer = optim.Adagrad(model.parameters(),
                                  lr=new_lr,
                                  lr_decay=lr_decay,
                                  weight_decay=wd)
    return optimizer


if __name__ == '__main__':
    main()
