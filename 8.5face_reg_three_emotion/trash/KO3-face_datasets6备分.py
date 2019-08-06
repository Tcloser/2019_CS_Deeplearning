from __future__ import print_function
from PIL import Image
import torch
from   torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as data
import numpy as np

import cv2
import os
import utils
from resnet_model.resnet import  resnet18

unloader = transforms.ToPILImage()
# ok
class Studentsface_Dataset(data.Dataset):
    """
        Smart_car Dataset.
    """
    def __init__(self,transform=None):

        self.transform = transform

    def __getitem__(self, index):# 根据数据集大小进行 整数 索引
        image = cv2.imread("/home/dooncloud/桌面/practice/8.5face_reg_three_emotion/train_pic/home_of_allpics/" + str(index+1) + ".jpg")   
        image = (cv2.resize(image,(224,224),interpolation=cv2.INTER_CUBIC) - 128.)/128. 
        image = image.swapaxes(1, 2).swapaxes(0, 1)
        image = torch.tensor(image, dtype=torch.float)
        image = image.type(torch.FloatTensor)
        #print(image.shape) #这时候还没有加上图片数量 ([3,224,224]) tensor
        if   index < 284:
            target = 0
        elif index < 674:
            target = 1
        else:
            target = 2
        return image, target

    def __len__(self):
        return 1077

def train(save_path, pretrained_path):
    #data_process
    print('==> Preparing data..')
    transform = transforms.Compose(
        [transforms.ToTensor(),
        ])

    trainset = Studentsface_Dataset(transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=5,  #应该扩充了维度([20，3，224，224])
                                            shuffle=True, num_workers=2)
    
    #print(trainloader)
    #resnet_constructor
    model = resnet18(num_classes=3)
    #model = nn.DataParallel(model)#key 并行计算
    
    #optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    current_lr = 0.0001

    # Assuming that we are on a CUDA machine, this should print a CUDA device:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    use_cuda = torch.cuda.is_available()#返回一个ture
    if use_cuda:
        model.to(device)
        criterion.to(device)

    #load_pretrained_model 暂时用不到
    epoch_ = 0
    if pretrained_path != "":
        epoch = load_pretrained_model(model, optimizer, pretrained_path)
        print(epoch + 1)
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    #train
    for epoch in range(5):  # loop over the dataset multiple times
        
        # running_loss = 0.0
        # running_acc = 0.0

        correct = 0
        train_loss = 0
        total = 0
        print(epoch)
        model.train()
        print('learning_rate: %s' % str(current_lr))
        for i, data in enumerate(trainloader):
            # get the inputs
            inputs, labels = data
            if use_cuda:
                inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            inputs, labels = Variable(inputs), Variable(labels)#自动求导
            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            utils.clip_gradient(optimizer, 0.001)#
            optimizer.step()
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            
            total += labels.size(0)
            correct += predicted.eq(labels.data).cpu().sum().item()

            utils.progress_bar(i, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (train_loss/(i+1), 100.*correct/total, correct, total))
        #     _1, _2 = torch.max(outputs.data, 1)

            
        #     for iii in range(len(labels)):
        #         if labels[iii] == _2[iii]:
        #             running_acc = running_acc + 1
        #             #print _2
        #     #if i % 200 == 199:    # print every 2000 mini-batches
        #     print('[%3d, %5d] loss: %.8f acc:%.8f' %
        #         (epoch + 1, i + 1, running_loss / (18*200.),running_acc/(18*200.)))
        #     running_loss = 0.0
        #     running_acc = 0.0
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            }, save_path)
    
    print('Finished Training')

def main():
    pass

if __name__ == '__main__':
    train(save_path="/home/dooncloud/桌面/practice/8.5face_reg_three_emotion/train_pic/tmp/Studentsface.t7",pretrained_path = "")