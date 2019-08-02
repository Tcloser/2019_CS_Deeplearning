import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import cv2
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

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
        out_bn = self.bn(out_conv1)
        out_relu = self.relu(out_bn)

        # out = self.conv2(x)
        # out = self.bn(out)
        # out = out.view(out.size(0), -1)
        # out = self.fc(out)
        return out_conv1,out_bn,out_relu

def main():
    # rand_image = torch.randn(48,48,3)
    # print(rand_image.shape)
    '''进图'''
    image = cv2.imread('/home/dooncloud/桌面/practice/7.30convtest/1.jpg')# to ndarray
    print(image.shape)

    net = resnet(3,64,kernel_size=3,stride=2,padding=1)

    image = (cv2.resize(image,(48,48),interpolation=cv2.INTER_CUBIC) - 128.)/128. #线性差值 归一化 (48, 48, 3)
    # 归一化后的图片看一看
    plt.subplot(2,2,1)
    plt.imshow(image)
    image = image.swapaxes(1, 2).swapaxes(0, 1)[np.newaxis, :]             #  **图像转置
    image = torch.tensor(image, dtype=torch.float, device=device)              
    image = image.type(torch.FloatTensor) # 转为float类型
    #print(image.shape)                                                    # torch.Size([1, 3, 48, 48])
    # pic_tensor3 = torch.from_numpy(pic_ndarray)                          # to Tensor   48*48*3  还需要数量维度 四个维度传入net
    # print(pic_tensor3.shape)
    # pic = pic_tensor3.expand(-1,-1,-1,1)
    # print(pic.shape)
    
    conv1_conv1,conv1_bn,conv1_relu = net.forward(image)                   # torch.Size([1, 64, 24, 24])
    
    '''图像'''
    conv1_conv1 = conv1_conv1.squeeze(0)                                   # torch.Size([64, 24, 24])
    tmpconv1 = unloader(conv1_conv1[63])
    plt.subplot(2,2,2)
    plt.imshow(tmpconv1)

    conv1_bn = conv1_bn.squeeze(0)                                         # torch.Size([64, 24, 24])
    tmpbn = unloader(conv1_bn[63])
    plt.subplot(2,2,3)
    plt.imshow(tmpbn)

    conv1_relu = conv1_relu.squeeze(0)
    #print(conv1_relu.shape)                                             # torch.Size([64, 24, 24])
    tmprelu = unloader(conv1_relu[63])
    plt.subplot(2,2,4)
    plt.imshow(tmprelu)

    '''显示'''
    plt.show()
    # tmp = np.asarray(tmp)
    # print(tmp.shape)                                                            # (24, 24)
    #ResultToJpg(pic)
    #print(type(img_list))

if __name__ == '__main__':
    unloader = transforms.ToPILImage()
    device = torch.device('cuda:0')
    main()