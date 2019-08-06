'''
概述：将筛选过后的文件夹smile\hand_on_face\half_face\normal中图片按照label中顺序重新命名为(int.jpg)存放到home_pic中，同时给出不同文件夹图片序号范围，便于后面读取添加标签
'''
import cv2
import glob # 用于读取文件夹下所有文件
import numpy
from PIL import Image
import os


'''参数初始化'''
def para_init():
    global label,label_sum,Path_of_the_store_of_allimagesfloders,put_allpics_into_home

    label = ["smile",'hand_on_face','half_face','normal'] # 更改标签即可将这几个文件夹中的图片放入home_of_allpics
    label_sum = []#存放图片数量
    Path_of_the_store_of_allimagesfloders = '/home/dooncloud/桌面/practice/8.5face_reg_three_emotion/train_pic/'
    put_allpics_into_home = "/home/dooncloud/桌面/practice/8.5face_reg_three_emotion/train_pic/home_of_allpics/"

'''求单个文件夹下图片总数'''
def sum_all_the_pic(which_one):                                            # 图片是从边缘提取之后筛选出来的
    path_file_number = glob.glob(Path_of_the_store_of_allimagesfloders+which_one+'/*')
    return len(path_file_number)

'''多个文件夹中图片按顺序存储到home_pic中'''
def save_pics(sum,which_one,start_numb_b):
    i = 0
    for name in glob.glob(Path_of_the_store_of_allimagesfloders+which_one+'/*'):
        i +=1 #first pic
        pic = cv2.imread(name)
        cv2.imwrite(put_allpics_into_home+str(start_numb_b)+'.jpg',pic)
        start_numb_b += 1 
    
def main():
    para_init()
    j = 1 # store the numb
    start_numb = 1 # start of different kind of emotion 
    for which_one in range(4):
        sum_pics = sum_all_the_pic(label[which_one])
        label_sum.append(sum_pics)
        print(label[which_one]+' include:'+str(sum_pics))
        # print(label_sum)
        save_pics(sum_pics,label[which_one],start_numb)
        start_numb = start_numb+label_sum[which_one]
        
        print(str(j)+'--->'+str(start_numb-1))
        j += label_sum[which_one]

if __name__ == "__main__":
    
    main()
    print('将上面区间填入3-face_datasets.py中即可')
