import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import cv2
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from torch.autograd import Variable

class resnet():
    def __init__(self, inplanes, planes, kernel_size,stride=1,padding=1):
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size,stride=stride,padding = padding)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size,stride=stride,padding = 1)
        self.bn   = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(2, 2)
        self.dropout10 = nn.Dropout(0.1)
        self.fc = nn.Linear(64, 5) 

    def forward(self, x):
        out_conv1 = self.conv1(x)                                  #torch.Size([1, 64, 24, 24])
        out_bn1 = self.bn(out_conv1)
        out_relu1 = self.relu(out_bn1)

        out_conv2 = self.conv2(out_relu1)
        print(out_conv2.shape)
        out_bn2 = self.bn(out_conv2)
        out_relu2 = self.relu(out_bn2)
        # out = out.view(out.size(0), -1)
        # out = self.fc(out)
        return out_conv1,out_bn1,out_relu1,out_conv2,out_bn2,out_relu2

def show_all(conv1_conv1,conv1_bn1,conv1_relu1,conv2_conv2,conv2_bn2,conv2_relu2):
    '''图像'''
    conv1_conv1 = conv1_conv1.squeeze(0)                                   # torch.Size([64, 24, 24])
    tmpconv1 = unloader(conv1_conv1[63])
    plt.subplot(3,3,4)
    plt.imshow(tmpconv1)

    conv1_bn1 = conv1_bn1.squeeze(0)                                         # torch.Size([64, 24, 24])
    tmpbn = unloader(conv1_bn1[63])
    plt.subplot(3,3,5)
    plt.imshow(tmpbn)

    conv1_relu1 = conv1_relu1.squeeze(0)                                # torch.Size([64, 24, 24])
    tmprelu = unloader(conv1_relu1[63])
    plt.subplot(3,3,6)
    plt.imshow(tmprelu)

    conv2_conv2 = conv2_conv2.squeeze(0)                                   # torch.Size([64, 24, 24])
    tmpconv2 = unloader(conv2_conv2[0])
    plt.subplot(3,3,7)
    plt.imshow(tmpconv2)

    conv2_bn2 = conv2_bn2.squeeze(0)                                         # torch.Size([64, 24, 24])
    tmpbn = unloader(conv2_bn2[0])
    plt.subplot(3,3,8)
    plt.imshow(tmpbn)

    conv2_relu2 = conv2_relu2.squeeze(0)
    #print(conv1_relu.shape)                                             # torch.Size([64, 24, 24])
    tmprelu = unloader(conv2_relu2[0])
    plt.subplot(3,3,9)
    plt.imshow(tmprelu)

def main():
    '''随机进图'''
    rand_image = torch.randn(48,48,3)

    rand_image = rand_image.data.numpy()

    image = rand_image
    # 适用于随机生成图
    # plt.subplot(3,3,2)
    # plt.imshow(image)
    '''cv2进图'''
    image = cv2.imread('/home/dooncloud/桌面/7.30convtest/1.jpg')# to ndarray
    #print(image.shape)

    net = resnet(3,64,kernel_size=3,stride=2,padding=1)

    image = (cv2.resize(image,(48,48),interpolation=cv2.INTER_CUBIC) - 128.)/128. #线性差值 归一化 (48, 48, 3)
    print(image.shape)
    # 适用于CV2进图
    plt.subplot(3,3,2)
    plt.imshow(image)

    image = image.swapaxes(1, 2).swapaxes(0, 1)[np.newaxis, :]             #  **图像转置
    # print(type(image))
    # print(image.shape)
    image = torch.tensor(image, dtype=torch.float, device=device)      
    # print(type(image))
    # print(image.shape)        
    image = image.type(torch.FloatTensor) # 转为float类型
    # print(type(image))
    # print(image.shape)
    '''卷积'''
    conv1_conv1,conv1_bn1,conv1_relu1,conv2_conv2,conv2_bn2,conv2_relu2 = net.forward(image)                   # torch.Size([1, 64, 24, 24])
    '''图像'''  
    show_all(conv1_conv1,conv1_bn1,conv1_relu1,conv2_conv2,conv2_bn2,conv2_relu2)
    '''显示'''
    plt.show()

if __name__ == '__main__':
    unloader = transforms.ToPILImage()
    device = torch.device('cuda:0')
    main()
