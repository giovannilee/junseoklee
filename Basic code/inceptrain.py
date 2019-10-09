import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision
import torchvision.datasets as dset
import matplotlib.pyplot as plt
import os
import numpy as np
from PIL import Image
import collections
# Model options
from torch.autograd import Variable
import torch.optim as optim
from inceptionv4 import InceptionV4
dataroot = 'train'
epochs = 15
embedding_size = 256
batch_size = 4

lr = 0.1
lr_decay = 1e-4
wd = 0.0
opt = 'adam'
gpu_id = '0'
seed = 0

os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
np.random.seed(seed)


class Scale(object):
    def __init__(self, size, interpolation=Image.BILINEAR):
        assert isinstance(size, int) or (isinstance(
            size, collections.lterable) and len(size) == 2)
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img):
        if isinstance(self.size, int):
            w, h = img.size

            if w <= h:
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
    transforms.Normalize(mean=[0.5],
                         std=[0.5])
])
train_dir = dset.ImageFolder(root='train', transform=transform)
train_loader = torch.utils.data.DataLoader(train_dir,
                                           batch_size=batch_size,
                                           shuffle=True)


def main():

    model = InceptionV4(num_classes=10)
    model.cuda()
    optimizer = create_optimizer(opt, model, lr)

    print('train epochs : {}'.format(epochs))
    count = 0

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0
        for batch_idx, data in enumerate(train_loader):
            img = data[0]
            label = data[1]

            img = img.cuda()
            label = label.cuda()
            img = Variable(img)
            label = Variable(label)
            #print('label:', label)
            cls = model(img)
            criterion = nn.CrossEntropyLoss()
            cross_entropy_loss = criterion(cls, label)
            loss = cross_entropy_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss = train_loss + loss.item()
            loading = 100 * (batch_idx / len(train_loader))
            print('loading... : {}%\r'.format(round(loading, 1)), end='')
        torch.save(model.state_dict(),
                   'checkpoints/checkpoint_{}.pt'.format(epoch))
        print('{} epochs done'.format(epoch))
        print('train_loss : ', train_loss)


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
