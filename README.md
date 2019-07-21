# 2019_CS_Deeplearning
 2019暑假在北京中科院实习经历


![ResNet50流程图](image/ResNet.JPG);

/---7.9周三---/     
'''argparse 命令行参数解析包：同时也将opt.model等赋初值 可以直接调用'''
import argparse #命令行参数解析
parser = argparse.ArgumentParser(description='PyTorch Fer2013 CNN Training')
parser.add_argument('--model', type=str, default='VGG19', help='CNN architecture')
parser.add_argument('--dataset', type=str, default='FER2013', help='CNN architecture')
parser.add_argument('--homepath', type=str, default='/home/dooncloud/桌面/practice/7.8Affectmet/Facial-Expression-Recognition/', help='CNN architecture')
parser.add_argument('--bs', default=50, type=int, help='learning rate')
parser.add_argument('--lr', default=0.0001, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
opt = parser.parse_args()
path = os.path.join(opt.homepath+opt.dataset + '_' + opt.model)

'''h5py 数据集的读取和写入'''csdn基本操作

import h5py
datafile = h5py.File(datapath, 'w')
datafile.create_dataset("Training_pixel", dtype = 'uint8', data=Training_x)
datafile.close()

'''dlib 可以用来68点人脸识别并裁减的机器学习库'''

import dlib
使用正向人脸检测可以提取人脸外部矩形框 使用训练好的68点特征检测可以进行人脸面部轮廓特征提取
HOG方向梯度直方图+分类器  基于边缘检测sobel算子

项目实现过程：
1、将数据集标签以及图像信息写入fer2013_vgg19.csv文件中，数据进行了人工分类加标签
2、preprocess.py实现数据预处理功能，通过调用h5py、csv包将数据从excel中读取出来并将其以列表形式保存在.h5文件的中，即/data下的data.h5，得到如下信息：
48*48像素的图片，像素点共有2304个，row；数据集其中28709个样本作为训练集,其中涉及excel数据读取、numpy下数据处理、h5py类型数据保存操作
(28709, 2304)
(3589, 2304)
(3589, 2304)
3、调用fer.py对上面预处理的数据进行提取，。torch中 (import torch.utils.data as data)调用data = h5py.File()即可

/---7.10周四---/      

1、测试ResNet网络(残差网络) 其为微软提出的比VGG网络更加适应深度更深(更复杂，参数多)的训练网络。显存占用率更高
传统的卷积网络在全连接网络和信息传递的时候会有信息损耗和丢失的问题，ResNet通过简化学习目标和难度，解决这一问题。
2、卷积神经网络中的卷积概念：在输入上进行特征的提取操作，对上一层特征进行提取操作。
35*35*3的原始图像(————6个5*5*3的filter————>)30*30*6的图像(————10个5*5*6的filter————>)25*25*10的图像……
类似于这样将原始图像细分为更多的小图，图像深度一直在增加，像素点在减小。
卷积是通过原图像像素点与fliter矩阵对应位置相乘后相加+偏置项，从而实现--通过不断地对输入图片进行不断地提取和压缩，最终得到高层次的特征。
--------www.jianshu.com/p/976dc5f82366      (7*7*3)conv(3*3*3*2)--->(3*3*2)

神经网络 神经元与变量之间的有限连接能够有效减小低相关的参数，卷积过后池化层是通过下采样，又是对连接参数的进一步缩小，这样大量不重要的参数将被舍去
stride步长
pad外部加零 增加边缘数据量防止每次卷积重叠部分过多，边缘效果凸显不出来。

3、目前常用神经网络 ：VGG--2013 、ResNet--2015 、

4、项目基本步骤
裁剪抠图：
（1）明确任务目标：准确识别视频中每一个人并且识别其动作行为和脸部表情
（2）首先与校方对接获取其摄像头监控视频资源
（3）对视频中每一帧图片手动标框(耗时长)，使用____对视频中标框目标进行裁剪后保存。将图片投入神经网络训练，以此来提高不标框时算法对视频中学生识别准确率
行为识别：
（4）训练完成后对多个视频进行大量裁剪，并对所裁剪的图片不符合条件的部分进行人工筛除。
（5）根据所要识别的动作进行分类打标签(耗时长)，比如低头、外头、抬头……等动作，人为归类完成后将其放入神经网络再一次训练，并取测试集数据进行对比验证
表情识别部分：
（6）重复（4）
（7）使用（5）训练的结果对图片进行筛检并打标签(耗时长)，只保留正面的多种表情图片，之后投入神经网络训练。

/---7.18周四---/
今天详细学习了一下ResNet残差网络结构 具体代码也更新了
/---7.21周日---/
打印了几份CS英文论文，以何凯明2015发表的NesNet论文为主。有时间更新一下Github还是很有必要的。
