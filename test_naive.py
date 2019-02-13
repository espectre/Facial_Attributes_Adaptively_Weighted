import torch
import torch.nn as nn
import torchvision.transforms as transforms
from customLayer.celebA import celeba
from model.resnet import resnet101
import os
from torch.autograd import Variable
import argparse
import csv

transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])
parser = argparse.ArgumentParser()
parser.add_argument('--workers', type=int, default=2)
parser.add_argument('--batchSize', type=int, default=32)
parser.add_argument('--gpu', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
parser.add_argument('--model', type=str,default='ckp/model.pth')
parser.add_argument('--attNum', type=int,default=1)
opt = parser.parse_args()
print(opt)

os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu
testset = CelebA('./datai/celebA/list_eval_partition.txt', './data/celebA/list_attr_celeba.txt', '2',
                  './data/celebA/img_align_celeba/', transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=opt.batchSize, shuffle=False, num_workers=opt.workers)

if not os.path.exists(opt.model):
    print('model doesnt exits')
    exit(1)

model=resnet50(pretrained=True)
model.fc=nn.Linear(2048,40)
model.load_state_dict(torch.load('ckp/model.pth'))
model.cuda()

model.eval()
correct = torch.FloatTensor(40).fill_(0)
total = 0
img=[]
with torch.no_grad():
    for batch_idx, (images, attrs,imgs_root) in enumerate(testloader):
        images = Variable(images.cuda())
        attrs = Variable(attrs.cuda()).type(torch.cuda.FloatTensor)
        output = model(images)
        com1 = output > 0
        com2 = attrs > 0
        for i in range(com1.shape[0]):
            if com1[i][opt.attNum-1]!=com2[i][opt.attNum-1]:
                #print("-------------------",imgs_root[i])
                img_elem=imgs_root[i].split('.')[1].split('/')[-1]
                img.append(str(img_elem))
        correct.add_((com1.eq(com2)).data.cpu().sum(0).type(torch.FloatTensor))
        total += attrs.size(0)
print("len:",len(img))
print(img)
print(correct / total)
print(torch.mean(correct / total))

err_csv=os.path.join(os.path.expanduser('.'),'deploy',str(opt.attNum)+'_error.csv')

with open(err_csv,'w',newline='') as f:
    writer=csv.writer(f,delimiter=',')
    for row in img:
        writer.writerow((str(row),))
