# -*- coding:utf-8 -*-
# get data and label from FER2013 save as image
# labels: 0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral
import csv
import os
import numpy as np
import pandas as pd
from PIL import Image
import itertools
from skimage import io

file = '/home/dooncloud/桌面/practice/7.18读取csv像素图/emotion_'
file_csv = '/home/dooncloud/桌面/practice/7.18读取csv像素图/'
all_the_number_is = 28070

# search all the pics in dir
def get_imlist(path):
    return [os.path.join(path,f) for f in os.listdir(path) if f.endswith('.jpg')]

'''没有调试ok'''

def read_pic(sum_pics):
    for i in range(sum_pics):
        temp_list = []
        try:
            img = io.imread(file+'0/'+str(i)+'.jpg')  # 打开图像
            img_ndarray = np.asarray(img)#numpy.ndarray
            img_list = img_ndarray.tolist()#list
            img_array = np.array(img_list).reshape(1,2304)
            temp_list.append(img_array.tolist())
            

        except FileNotFoundError:
            continue
        except ValueError:
            continue
    
    I = np.asarray(temp_list)
    print(I)
    return I
    #img_ndarray = np.asarray(img, dtype='float64') 


if __name__ == "__main__":
    
    # for emotion_num in range(0,7):
    #     c = get_imlist(file+str(emotion_num))
    #     print(len(c))    # 图像个数

    emotion_num =0
    pic_sum_number = get_imlist(file+str(emotion_num))
    print(len(pic_sum_number))    # 图像个数

    img = io.imread(file+'0/1.jpg')
    print(type(img))

    #img = read_pic(all_the_number_is)     #此处调用函数
    
    img_ndarray = np.asarray(img)#numpy.ndarray
    print(type(img_ndarray))

    img_ndarray = img_ndarray.tolist()#list
    print(type(img_ndarray))

    img = np.array(img_ndarray).reshape(1,2304) #48*48 list--> 1*2304 numpy.ndarray
    print(img)

    save = pd.DataFrame(img)#pandas.core.frame.DataFrame
    print(save)

    save.to_csv('/home/dooncloud/桌面/practice/7.18读取csv像素图/emotion_0.csv', index=False, header=False)