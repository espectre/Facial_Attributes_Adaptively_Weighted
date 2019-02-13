import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.nn import init
import torchvision.utils as vutils
from customLayer.celebA import celeba
from model.resnet import resnet101, resnet101_full
import itertools
import cv2
import os
from torch.autograd import Variable
import argparse
import numpy as np
from torch.optim.lr_scheduler import *

transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])
parser = argparse.ArgumentParser()
parser.add_argument('--workers', type=int, default=2)
parser.add_argument('--batchSize', type=int, default=32)
parser.add_argument('--gpu', type=str, default='4', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
parser.add_argument('--model', type=str)
opt = parser.parse_args()
print(opt)

os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu
testset = celeba('./data/celebA/list_eval_partition.txt', './data/celebA/list_attr_celeba.txt', '2',
                  './data/celebA/img_align_celeba/', transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=opt.batchSize, shuffle=False, num_workers=opt.workers)

#if not os.path.exists(opt.model):
#    print('model doesnt exits')
#    exit(1)
resnet = resnet101_full(opt.model, num_classes=40)
resnet.cuda()

resnet.eval()
correct = torch.FloatTensor(40).fill_(0)
total = 0
for batch_idx, (images, attrs) in enumerate(testloader):
    images = Variable(images.cuda())
    attrs = Variable(attrs.cuda()).type(torch.cuda.FloatTensor)
    output = resnet(images)
    com1 = output > 0
    com2 = attrs > 0
    correct.add_((com1.eq(com2)).data.cpu().sum(0).type(torch.FloatTensor))
    total += attrs.size(0)
print(correct / total)
print(torch.mean(correct / total))
