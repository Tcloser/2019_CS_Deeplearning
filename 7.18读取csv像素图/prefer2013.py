# -*- coding:utf-8 -*-
# get data and label from FER2013 save as image
# labels: 0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral
import csv
import os
import numpy as np
from PIL import Image
import itertools
from skimage import io

file = '/home/dooncloud/桌面/practice/7.18读取csv像素图/fer2013.csv'

def createFile(filePath):
    if os.path.exists(filePath):
        print('exist!')
    else:
        os.makedirs(filePath)

for i in range(0,7):
    filepath = '/home/dooncloud/桌面/practice/7.18读取csv像素图/emotion_'+str(i)
    createFile(str(filepath))

# Creat the list to store the data and label information
Training_x = []
Training_y = []
PublicTest_x = []
PublicTest_y = []
PrivateTest_x = []
PrivateTest_y = []

i = 0
with open(file,'r') as csvin:
    data=csv.reader(csvin)
    for row in data:     # 对训练集与测试集进行分类
        
        if row[2] == 'Training':# C列
            temp_list = []
            for pixel in row[1].split( ):#对字符串进行切片 提取出数字便于进行转换
                temp_list.append(int(pixel))#numpy append 添加新的对象
            I = np.asarray(temp_list) #I是ndarray格式
            #print(int(row[0]))
            #print(np.size(I.tolist()))
            pic = I.tolist()#一维数组
            pic = np.array(pic).reshape(48,48)
            #pic = pic.tolist()#array to list
            #print(len(pic))#48维度
            try:
                io.imsave('/home/dooncloud/桌面/practice/7.18读取csv像素图/emotion_'+str(int(row[0]))+'/'+str(i)+'.jpg',pic)
                i += 1
            except ValueError:
                #i += 1
                continue
            # #存储
            # Training_y.append(int(row[0]))
            # Training_x.append(I.tolist())#像素点列表格式存储
            
        else:
            None
            i += 1
         