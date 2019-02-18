import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.nn import init
import torchvision.utils as vutils
from customLayer.celebA import celeba
from model.resnet import resnet101
import itertools
import cv2
import os
from torch.autograd import Variable
import argparse
import numpy as np
from torch.optim.lr_scheduler import *

transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])


transform_val = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

parser = argparse.ArgumentParser()
parser.add_argument('--workers', type=int, default=2)
parser.add_argument('--batchSize', type=int, default=32)
parser.add_argument('--niter', type=int, default=10, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate, default=0.0002')
parser.add_argument('--gpu', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
opt = parser.parse_args()
print(opt)

os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu

trainset = celeba('./data/celebA/list_eval_partition.txt', './data/celebA/list_attr_celeba.txt', '0',
                  './data/celebA/img_align_celeba/', transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=opt.batchSize, shuffle=True, num_workers=opt.workers)

valset = celeba('./data/celebA/list_eval_partition.txt', './data/celebA/list_attr_celeba.txt', '1',
                  './data/celebA/img_align_celeba/', transform_val)
valloader = torch.utils.data.DataLoader(valset, batch_size=opt.batchSize, shuffle=True, num_workers=opt.workers)



resnet = resnet101(pretrained=True, num_classes=40)
resnet.cuda()
criterion = nn.MSELoss(reduce=False)
optimizer = optim.SGD(resnet.parameters(), lr=opt.lr, momentum=0.9, weight_decay=5e-4)
scheduler = StepLR(optimizer, step_size=3)
kk = 100
count = 0
weight = torch.FloatTensor(40).fill_(1.0).cuda()
print(type(weight))
val_loss = []

def validation():
    resnet.eval()
    for _, (images, attrs) in enumerate(valloader):
        images = Variable(images.cuda())
        attrs = Variable(attrs.cuda()).type(torch.cuda.FloatTensor)
        output = resnet(images)
        loss = criterion(output, attrs)
        loss_s = torch.mean(loss, dim=0)
        val_loss.append(loss_s.data.clone().cpu().numpy())
        global count
        count += 1
        if count >= 2*kk and count % kk == 0:
            # pre_val_mean 前面100个batch的loss均值
            # cur_val_mean 当前100个batch的loss均值
            pre_val_mean = np.mean(val_loss[(count-2*kk):(count-kk)], axis=0)
            cur_val_mean = np.mean(val_loss[(count-kk):count], axis=0)
            # trend loss的变化趋势
            trend = np.abs(cur_val_mean - pre_val_mean) / cur_val_mean
            norm_trend = trend / np.mean(trend)
            norm_loss = cur_val_mean / np.mean(cur_val_mean)
            weight1 = norm_loss * norm_trend
            global weight
            weight = Variable(torch.from_numpy(weight1 / np.mean(weight1))).cuda()
            print(weight)
        break

def train(epoch):
    print('\nEpoch: %d' % epoch)
    scheduler.step()
    for batch_idx, (images, attrs) in enumerate(trainloader):
        resnet.train()
        images = Variable(images.cuda())
        attrs = Variable(attrs.cuda()).type(torch.cuda.FloatTensor)
        optimizer.zero_grad()
        output = resnet(images)
        loss = criterion(output, attrs)
        #loss.backward(torch.ones(1).cuda())
        loss_s = torch.mean(loss, dim=0)
        loss_s.backward(weight)
        optimizer.step()

        print('[%d/%d][%d/%d] loss: %.4f' % (epoch, opt.niter, batch_idx, len(trainloader), loss_s.mean()))
        validation()



def test(epoch):
    print('\nTest epoch: %d' % epoch)
    resnet.eval()
    correct = torch.FloatTensor(40).fill_(0)
    total = 0
    with torch.no_grad():
        for batch_idx, (images, attrs) in enumerate(valloader):
            images = Variable(images.cuda())
            attrs = Variable(attrs.cuda()).type(torch.cuda.FloatTensor)
            output = resnet(images)
            com1 = output > 0
            com2 = attrs > 0
            correct.add_((com1.eq(com2)).data.cpu().sum(0).type(torch.FloatTensor))
            total += attrs.size(0)
    print(correct / total)
    print(torch.mean(correct / total))



for epoch in range(0, opt.niter):
    train(epoch)
    test(epoch)
torch.save(resnet.state_dict(), 'ckp/resnet.pth')
