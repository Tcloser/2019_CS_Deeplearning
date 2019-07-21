'''参考
https://zhuanlan.zhihu.com/p/25572330 知乎
https://www.cnblogs.com/Jerry-Dong/p/8109938.html COFAR-10数据集的使用说明
https://blog.csdn.net/shi2xian2wei2/article/details/84308644 卷积层的优化
https://blog.csdn.net/u014380165/article/details/79058479 batch类型数据封装成tensor然后再包装成variable
---------------------7.2第一次跑过--------------------------
---------------------此代码参考下文--------------------------
https://blog.csdn.net/shi2xian2wei2/article/details/84308644
https://www.cnblogs.com/how-chang/p/9576956.html 卷积神经网络及实现步骤
https://blog.csdn.net/sunflower_sara/article/details/81322048 神经网络中三种池化层的作用
https://www.cnblogs.com/qinduanyinghua/p/9311410.html pythrch save 用法
https://blog.csdn.net/qq_42393859/article/details/84336558 pytorch处理基本步骤 
    
    加载数据并生成batch数据--已生成
    数据预处理 --将batch数据导入
    构建神经网络 --全卷积CNN
    Tensor和Variable 1
    定义loss 1
    自动求导
    优化器更新参数
    训练神经网络
    参数_定义
    参数_初始化
    如何在训练时固定一些层？
    绘制loss和accuracy曲线
    torch.nn.Container和torch.nn.Module
    各层参数及激活值的可视化
    保存训练好的模型
    如何加载预训练模型
    如何使用cuda进行训练

注意点：
    1、torchvision.datasets中有几个已经定义好的数据集类 这里的CIFAR10即为一种抽象子类
        torchvision.datasets.MNIST类：标签是一维的，不是one-hot稀疏标签。
        torchvision.datasets.CIFAR10
        torchvision.datasets.ImageFolder
    2、这里没有定义torch.utils.data.Dataset的子类  即两个函数是__len__和__getitem__
    3、！还不懂：mean pooling比较容易让人理解错的地方就是会简单的认为直接把梯度复制N遍之后直接反向传播回去，但是这样会造成loss之和变为原来的N倍，网络是会产生梯度爆炸的。
        max pooling也要满足梯度之和不变的原则，max pooling的前向传播是把patch中最大的值传递给后一层，而其他像素的值直接被舍弃掉。那么反向传播也就是把梯度直接传给前一层某一个像素，而其他像素不接受梯度，也就是为0。所以max pooling操作和mean pooling操作不同点在于需要记录下池化操作时到底哪个像素的值是最大
    （1）邻域大小受限造成的估计值方差增大；mean pool
    （2）卷积层参数误差造成估计均值的偏移。max pool    这两个能够有效防止交叉影响
    4、http://scs.ryerson.ca/~aharley/vis/conv/flat.html 神经网络可视化网站
'''
import torch
import torchvision
import torchvision.transforms as transforms
'''-------------------------------------step1 数据预处理'''
transform = transforms.Compose(
     [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
#原为从网上下载数据集 直接下载解压后修改存放地址即可跳过下载步骤
trainset = torchvision.datasets.CIFAR10('/home/dooncloud/桌面/practice/7.3_CIFAR-10/data/cifar-10-python', train=True,
                                        download=False, transform=transform)
#/home/dooncloud/桌面/实习代码/data/cifar-10-python为cifar-10-batches-py存放路径
testset = torchvision.datasets.CIFAR10('/home/dooncloud/桌面/practice/7.3_CIFAR-10/data/cifar-10-python', train=False,
                                       download=False, transform=transform)
# testset = torchvision.datasets.CIFAR10(root='./data/cifar-10-python', train=False,
#                                        download=False, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=1000,      
                                          shuffle=True, num_workers=3) 
'''
trainset：Dataset类型，从其中加载数据
batch_size：int，可选。每个batch加载多少样本
shuffle：bool，可选。为True时表示每个epoch都对数据进行洗牌
sampler：Sampler，可选。从数据集中采样样本的方法。
num_workers：int，可选。加载数据时使用多少子进程。默认值为0，表示在主进程中加载数据。
collate_fn：callable，可选。
pin_memory：bool，可选
drop_last：bool，可选。True表示如果最后剩下不完全的batch,丢弃。False表示不丢弃。
'''
testloader = torch.utils.data.DataLoader(testset, batch_size=1000,
                                         shuffle=False, num_workers=3)
classes = ('plane', 'car', 'bird', 'cat','deer', 'dog', 'frog', 'horse', 'ship', 'truck')#labels包括这几种

'''#code start
def imshow(img):
    img = img / 2 + 0.6 #unnormalize 不标准归一化
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)))
    #np.random.rand(1,2)生成随机0-1矩阵

dataiter = iter(trainloader)#show some random training images
images, labels = dataiter.next()#同时读取图片和标签 所以显示的图像都是一一对应的

imshow(torchvision.utils.make_grid(images))

print(' '.join('%5s'%classes[labels[j]] for j in range(10)))
plt.show()
'''#code end
'''-------------------------------------step2 定义全卷积层 效果好一些'''
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding = 1)
        self.conv2 = nn.Conv2d(64, 64, 3, padding = 1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding = 1)
        self.conv4 = nn.Conv2d(128, 128, 3, padding = 1)
        self.conv5 = nn.Conv2d(128, 256, 3, padding = 1)
        self.conv6 = nn.Conv2d(256, 256, 3, padding = 1)
        self.maxpool = nn.MaxPool2d(2, 2)#窗口大小是2*2
        self.avgpool = nn.AvgPool2d(2, 2)
        self.globalavgpool = nn.AvgPool2d(8, 8)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        self.dropout50 = nn.Dropout(0.5)
        self.dropout10 = nn.Dropout(0.1)
        self.fc = nn.Linear(256, 10)  #10输出层

    def forward(self, x):
        x = self.bn1(F.relu(self.conv1(x)))
        x = self.bn1(F.relu(self.conv2(x)))
        x = self.maxpool(x)
        x = self.dropout10(x)#正则化减少过拟合的情况

        x = self.bn2(F.relu(self.conv3(x)))
        x = self.bn2(F.relu(self.conv4(x)))
        x = self.avgpool(x)#只求平均
        x = self.dropout10(x)

        x = self.bn3(F.relu(self.conv5(x)))
        x = self.bn3(F.relu(self.conv6(x)))
        x = self.globalavgpool(x)
        x = self.dropout50(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
net = Net()


'''-------------------------------------step3 定义损失函数和优化器 adam对复杂函数的优化效果好一些'''
import torch.optim as optim#优化器包
criterion = nn.CrossEntropyLoss() 
#optimizer 这一关键参数能够保存当前参数以及更新梯度信息  参数直接调用.parameters即可 后面加上学习率
optimizer = optim.Adam(net.parameters(), lr=0.001)  #学习率0.001 adma优化率85%  SGD为55%

'''-------------------------------------step4 训练网络 '''
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net.to(device)#这里注意一下 训练的数据也要.to(device)

for epoch in range(10):

    running_loss = 0.
    batch_size = 1000# 10 100 1000不同参数时速度不一样
    
    for i, data in enumerate(
            torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                        shuffle=True, num_workers=2), 0):
        
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device) #放入GPU

        optimizer.zero_grad()#每次迭代清空上一次的梯度
        outputs = net(inputs)#对x进行预测
        loss = criterion(outputs, labels)#计算损失
        loss.backward()#损失反向传播
        optimizer.step()#更新梯度

        print('[%d, %5d] loss: %.4f' %(epoch + 1, (i+1) * batch_size, loss.item()))#[10, 49000] loss: 0.2587
print('Finished Training')
'''-------------------------------------step5 用下面的语句存储或读取保存好的模型：'''
torch.save(net, 'cifar10.pkl')#保存整个网络和参数和代码同一个文件夹下面
print('Finished saving')

#---------------------------------------------先调试好训练的部分-------------------------------------------------
'''-------------------------------------step6 测试训练结果'''
net = torch.load('cifar10.pkl')#重新加载模型

correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))

class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
with torch.no_grad():
    for data in testloader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1


for i in range(10):
    print('Accuracy of %5s : %2d %%' % (
classes[i], 100 * class_correct[i] / class_total[i]))
   
