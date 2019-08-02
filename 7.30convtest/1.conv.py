import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np

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
        out = self.conv1(x)
        out = self.bn(out)
        print(out.shape)
        # out = self.relu(out)
        # out = self.conv2(x)
        # out = self.bn(out)
        return out

def main():
    # rand_image = torch.randn(48,48,3)
    # print(rand_image.shape)
    image = cv2.imread('/home/dooncloud/桌面/7.30convtest/1.jpg')# to ndarray
    print(image.shape)


    #net = ResNet18()
    net = resnet(3,64,kernel_size=2,stride=1,padding=1)


    image = (cv2.resize(image,(48,48),interpolation=cv2.INTER_CUBIC) - 128.)/128. 
    image = image.swapaxes(1, 2).swapaxes(0, 1)[np.newaxis, :]                  #图像转置
    image = torch.tensor(image, dtype=torch.float, device=device)               # ([1,3,48,48])
    image = image.type(torch.FloatTensor) # 转为float类型
    print(image.shape)

    conv1_pic = net.forward(image)
    print(conv1_pic.shape)

if __name__ == '__main__':
    device = torch.device('cuda:0')
    main()