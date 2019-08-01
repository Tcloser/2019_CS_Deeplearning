'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from torch.autograd import Variable


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False) #步长外部传进来
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        '''
        downsample部分
        1、如果输入通道数与输出通道数不等时
        2、步长不为1时
        '''
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))# 经卷积 bn relu
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)    #与残差相加
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4#最后输出512*4 通道

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):#resnet1.0 basic原理
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):#一步一步往下看 卷积和池化的计算公式 H(W)=(H(in)+2*padding-kernel_size)/stride
    def __init__(self, block, num_blocks, num_classes=7):#ResNet(BasicBlock, [2,2,2,2]) 使用的basic类型和卷积层每一层运算次数ResNet18为[2，2，2，2]
        super(ResNet, self).__init__()
        self.in_planes = 64#输入通道数为64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)#二维卷积 输入3通道 输出64通道 卷积核3个 步长为1 padding为1补一圈 无偏置项
        self.bn1 = nn.BatchNorm2d(64)#卷积输出为64通道 经过BN-----》查BN是啥
        '''
        1、由num_blocks[]+stride 到def _make_layer()函数给每一层layer的每一次卷积赋stride值 第二个卷积开始stride=1
        注意：这里padding隐藏在BasicBlock中  如果计算中出现小数说明有隐藏padding未找到！！
        2、返回的nn.Sequential()由所选择的block类型回到上面basicblock（bottleblock）类调整输出图像的通道大小
        '''
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)#这个 layer4 维度可以输出
        self.avgpool = nn.AvgPool2d(6,stride=1) #平均池化要把最后输出 out = self.avgpool(out) out维度变为1*1*512           
        self.fc = nn.Linear(512,num_classes)#将512种类全连接到7种类型上 num_classes是最后目标要分成的7类
        self.output = 1
        '''def _make_layer()
        1、给不同深度的卷积层的每一层stride赋值   
        2、stride！=1 时 维度的变化在同一卷积层只改变一次'''
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)#(num_blocks-1)给这一层卷积的第一层之外的所有层stride置1 只有第一层接收传过来的stride
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):#x是输入的图片数据 数据集改变的是x
        out = F.relu(self.bn1(self.conv1(x)))# size/1 看
        out = self.layer1(out)# size/1
        out = self.layer2(out)# size/2 只在这一层的第一次运算改变图片大小
        '''
        print(net.output.shape) 可返回layer(n)卷积后维度的大小
        输出结果：
        layer3--->torch.Size([200, 256, 11, 11])..] | Loss: 1.756 | Acc: 25.000% (30/120)     
        layer4--->torch.Size([200, 512, 6, 6])..] | Loss: 1.756 | Acc: 25.000% (30/120)
        '''
        self.output = self.layer3(out)
        out = self.output
        out = self.layer4(out)# size/2
        #这里大小6*6 对
        #out = F.avg_pool2d(out, 4)# size-4)/6+1
        out = self.avgpool(out)# size-6+1       平均池化 跟卷积运算图像变化算法相同
        out = out.view(out.size(0), -1)
        out = F.dropout(out, p=0.5, training=self.training)#类似relu激活函数 
        out = self.fc(out)#basic block中是1*1*512---》1*1*7 bottle中是 1*1*512*expansion--》1*1*7
        return out #7种表情


def ResNet18():
    return ResNet(BasicBlock, [2,2,2,2])#每个layer里面有两个残差块
'''
输入图片为44*44*1
'''
