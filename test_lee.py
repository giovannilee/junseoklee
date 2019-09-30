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
from PIL import Image
from RGModel import RGModel
import random
import math
from collections import Counter
import pickle
import csv
from sklearn.metrics import confusion_matrix

test_dir = 'Newdata/test/shiftblur'
embedding_size = 256
dir_length = len(os.listdir(test_dir))
batch_size = 1


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


def np_split(x):
    x = np.array_split(x, 6)
    return x

def imcat(batch_idx, batch_size):
    im_storage = []
    start = (batch_idx)*(batch_size)
    for i in range(batch_size):
        im = Image.open("Newdata/test/shiftblur/{}.jpg".format(start+i))
        im = (transform(im)+1)*100
        im_storage.append(im)
    batched_im = np.stack(im_storage, axis=0)
    return_value = torch.from_numpy(batched_im)
    return return_value
def getlabel(batch_idx, batch_size):
    label_storage = []
    start = (batch_idx) * (batch_size)
    for i in range(batch_size):
        with open("Newdata/test/label/{}.pickle".format(start + i), "rb") as fr:
            label = pickle.load(fr)
        # label = np.argmax(label)
        label_storage.append(label)
    batched_label = np.stack(label_storage, axis=0)
    return_value = torch.from_numpy(batched_label)
    return return_value

transform = transforms.Compose([
    Scale(224),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    # transforms.Normalize(mean = [ 0.5, 0.5, 0.5 ],
    # std = [ 0.5, 0.5, 0.5 ])
])

test_loader = torch.utils.data.DataLoader(test_dir, batch_size=1, shuffle=False)
dir_length = len(os.listdir(test_dir))

loader_length = int((dir_length) / (batch_size))
model = RGModel(embedding_size, 60, pretrained=False)
model.eval()
model.cuda()
files = os.listdir('Newdata/test/shiftblur')

for k in range(1, 201):
    model.load_state_dict(torch.load('checkpoint/checkpoint_{}.pt'.format(k)))

    acc1 = 0
    acc2 = 0
    acc3 = 0
    acc4 = 0
    acc5 = 0
    acc6 = 0
    tot = 0

    cam1_pred_arr = []
    cam2_pred_arr = []
    cam3_pred_arr = []
    cam4_pred_arr = []
    cam5_pred_arr = []
    cam6_pred_arr = []

    cam1_true_arr = []
    cam2_true_arr = []
    cam3_true_arr = []
    cam4_true_arr = []
    cam5_true_arr = []
    cam6_true_arr = []

    print('-------------------------------------------')
    print('{} epoch result:'.format(k))

    for batch_idx in range(loader_length):
        img = imcat(batch_idx, batch_size)
        label = getlabel(batch_idx, batch_size)
        label_split = torch.split(label,10,dim=1)

        cam1 = torch.argmax(label_split[0]).cpu().numpy().item()
        cam2 = torch.argmax(label_split[1]).cpu().numpy().item()
        cam3 = torch.argmax(label_split[2]).cpu().numpy().item()
        cam4 = torch.argmax(label_split[3]).cpu().numpy().item()
        cam5 = torch.argmax(label_split[4]).cpu().numpy().item()
        cam6 = torch.argmax(label_split[5]).cpu().numpy().item()

        img = img.cuda()
        img = Variable(img)

        cls = model(img)
        cls_b = torch.split(cls, 10, dim=1)
        pred1 = torch.argmax(cls_b[0]).cpu().numpy().item()
        pred2 = torch.argmax(cls_b[1]).cpu().numpy().item()
        pred3 = torch.argmax(cls_b[2]).cpu().numpy().item()
        pred4 = torch.argmax(cls_b[3]).cpu().numpy().item()
        pred5 = torch.argmax(cls_b[4]).cpu().numpy().item()
        pred6 = torch.argmax(cls_b[5]).cpu().numpy().item()

        cam1_pred_arr.append(pred1)
        cam1_true_arr.append(cam1)

        cam2_pred_arr.append(pred2)
        cam2_true_arr.append(cam2)

        cam3_pred_arr.append(pred3)
        cam3_true_arr.append(cam3)

        cam4_pred_arr.append(pred4)
        cam4_true_arr.append(cam4)

        cam5_pred_arr.append(pred5)
        cam5_true_arr.append(cam4)

        cam6_pred_arr.append(pred6)
        cam6_true_arr.append(cam5)

        #print(cam1,cam2,cam3,cam4,cam5,cam6)
        #print(pred1,pred2,pred3,pred4,pred5,pred6)
        #print('-----------------------------------')

        if pred1 == cam1:
            acc1 = acc1 + 1
        if pred2 == cam2:
            acc2 = acc2 + 1
        if pred3 == cam3:
            acc3 = acc3 + 1
        if pred4 == cam4:
            acc4 = acc4 + 1
        if pred5 == cam5:
            acc5 = acc5 + 1
        if pred6 == cam6:
            acc6 = acc6 + 1
        loading = 100*(batch_idx/loader_length)
        print('loading... : {}%\r'.format(round(loading,1)),end='')

    print('cam1 accuracy : {}'.format(100 * (acc1 / len(files))))
    print('cam2 accuracy : {}'.format(100 * (acc2 / len(files))))
    print('cam3 accuracy : {}'.format(100 * (acc3 / len(files))))
    print('cam4 accuracy : {}'.format(100 * (acc4 / len(files))))
    print('cam5 accuracy : {}'.format(100 * (acc5 / len(files))))
    print('cam6 accuracy : {}'.format(100 * (acc6 / len(files))))
    avg = np.average([acc1, acc2, acc3, acc4, acc5, acc6]) * 100 / len(files)
    print('average accuracy : {}'.format(avg))

'''
    conf_matrix1 = confusion_matrix(cam1_true_arr, cam1_pred_arr, labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    conf_matrix2 = confusion_matrix(cam2_true_arr, cam2_pred_arr, labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    conf_matrix3 = confusion_matrix(cam3_true_arr, cam3_pred_arr, labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    conf_matrix4 = confusion_matrix(cam4_true_arr, cam4_pred_arr, labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    conf_matrix5 = confusion_matrix(cam5_true_arr, cam5_pred_arr, labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    conf_matrix6 = confusion_matrix(cam6_true_arr, cam6_pred_arr, labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

    print(conf_matrix1)
    print(conf_matrix2)
    print(conf_matrix3)
    print(conf_matrix4)
    print(conf_matrix5)
    print(conf_matrix6)
'''






            
  
    
  
