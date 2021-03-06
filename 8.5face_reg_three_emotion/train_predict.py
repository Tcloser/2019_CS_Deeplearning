'''
8.8 添加 pr_work_pics(label_pr_work_folder)函数 
    return 文件夹中图片的数量列表 更新到dataset.len
'''
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
import glob
import cv2
import os
import utils
from resnet_model.resnet import  resnet18
import put_pics_into_home_andrename as pr_work

unloader = transforms.ToPILImage()

def para_init():
    global label_pr_work_folder,pics_nub_list,label_exp,para_batchsize,para_learningrate,train_pic_read_path,train_para_path,predict_read_path,predict_save_path
    pics_nub_list = []
    label_pr_work_folder = ["Fine",'ignore','tired','focus_on']# 手动分类的四个文件夹名(看起来是什么表现)

    label_exp =            ["Fine",'ignore','tired','focus_on']                 # 分类 最后图片predict出来的四种类型(我认为是什么学习态度)
    para_batchsize = 15                                              # in 5 pics each time
    para_learningrate = 0.0001                                      # 0.1 0.001 0.00001
    train_pic_read_path = "/home/dooncloud/桌面/practice/8.5face_reg_three_emotion/train_pic/home_of_allpics/" # 前面就已经保存为1.jpg类型了
    train_para_path = "/home/dooncloud/桌面/practice/8.5face_reg_three_emotion/train_pic/tmp/Studentsface.t7"  # 训练结果保存
    predict_read_path = "/home/dooncloud/桌面/person_face/pic/"        # 预测图片文件夹
    predict_save_path = "/home/dooncloud/桌面/person_face/predict/"  # 预测后图片另存

def pr_work_pics(label_pr_work_folder):
    global pics_nub_list                                                # 设置为全局变量 Studentsface_Dataset中调用
    pics_nub_list = pr_work.pic_deel(label_pr_work_folder)
    print('下面这四个数字将自动写入Studentsface_Dataset中，四分类显示各个文件夹中图片数量：')
    print(pics_nub_list[0],pics_nub_list[1],pics_nub_list[2],pics_nub_list[3])

class Studentsface_Dataset(data.Dataset):
    """
        Studentsface emotion Dataset.
    """
    def __init__(self,transform=None):

        self.transform = transform

    def __getitem__(self, index):# 根据数据集大小进行 整数 索引
        image = cv2.imread(train_pic_read_path + str(index+1) + ".jpg")   
        image = (cv2.resize(image,(224,224),interpolation=cv2.INTER_CUBIC) - 128.)/128. 
        image = image.swapaxes(1, 2).swapaxes(0, 1)
        image = torch.tensor(image, dtype=torch.float)
        image = image.type(torch.FloatTensor)
        #print(image.shape) #这时候还没有加上图片数量 ([3,224,224]) tensor
        if   index < pics_nub_list[0]:
            target = 0
        elif index < pics_nub_list[1]:
            target = 1
        elif index < pics_nub_list[2]:
            target = 2
        else:
            target = 3
        return image, target

    def __len__(self):
        return pics_nub_list[3]-1

def train(save_path, pretrained_path):
    #data_process
    print('==> Preparing data..')

    #pr_work_pics()#这个要返回一下几个数量

    transform = transforms.Compose(
        [transforms.ToTensor(),
        ])

    trainset = Studentsface_Dataset(transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=para_batchsize,  #应该扩充了维度([20，3，224，224]) 读取了dataset中的总长度__len__
                                            shuffle=True, num_workers=2)
    
    #print(trainloader)
    #resnet_constructor
    model = resnet18(num_classes=len(label_exp))
    #model = nn.DataParallel(model)#key 并行计算
    
    #optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=para_learningrate)
    current_lr = para_learningrate

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
    for epoch in range(15):  # loop over the dataset multiple times
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

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            }, save_path)
    
    print('Finished Training')

def load_pretrained_model(model, optimizer, pretrained_path):
    checkpoint = torch.load(pretrained_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    #model.eval()
    #model = torch.load(pretrained_path)
    return epoch

def predictor():
    # label_exp = ["smile",'handon_face',"half_face",'normal']
    # test_num = 0.
    # test_acc = 0.
    device = torch.device('cuda:0')
    model = torchvision.models.resnet18(num_classes=len(label_exp))
    checkpoint = torch.load(train_para_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    epoch_ = checkpoint['epoch']
    model.eval()
    model.to(device)
    #for i in range(21):
    i = 0
    for name in glob.glob(predict_read_path+'*'):
        # images in
        image = cv2.imread(name)
        out_image = image
        # 图片格式 np-->tensor
        image = (cv2.resize(image,(224,224),interpolation=cv2.INTER_CUBIC) - 128.)/128.
        image = image.swapaxes(1, 2).swapaxes(0, 1)[np.newaxis, :] #传入图片为一张的时候
        image = torch.tensor(image, dtype=torch.float, device=device)
        # predict
        outputs = model(image) #将单个图片进行归类预测
        outputs = outputs.cpu().clone()
        outputs_ = (outputs.detach().numpy())[0]
        tmp = 0
        for i in range(len(outputs_)):
            tmp = tmp + np.e**outputs_[i]
        for i in range(len(outputs_)):
            outputs_[i] = np.e**outputs_[i] / tmp
        print(outputs_)
        #curr_pred, curr_conf = max(enumerate(outputs[0]),key = len)
        _1, _2 = torch.max(outputs.data,1)
        print ("label:" + str(_2.item()))
        print ("标签:" + label_exp[_2.item()] + " 概率:" + str(outputs_[_2.item()]))
        # save
        cv2.imwrite(predict_save_path+label_exp[_2.item()]+str(outputs_[_2.item()])+'.jpg',out_image)

    print('Done！')

def main():
    para_init()
    '''这两个同步训练'''
    pr_work_pics(label_pr_work_folder)# 传入文件夹列表
    train(save_path=train_para_path,pretrained_path = "")
    '''单独'''
    predictor()

if __name__ == '__main__':
    main()