import cv2
import glob
import numpy
from PIL import Image
import os
'''将筛选过后的文件夹中图片按照顺序重新命名存放到home_pic'''

'''参数初始化'''
def para_init():
    global label_sum,Path_of_the_store_of_allimagesfloders,put_allpics_into_home
    #label = ["test_smile",'test_halfface','test_tired','test_normal'] # 更改标签即可将这几个文件夹中的图片放入home_of_allpics
    label_sum = []#存放图片数量
    Path_of_the_store_of_allimagesfloders = '/home/dooncloud/桌面/practice/8.5face_reg_three_emotion/train_pic/'
    put_allpics_into_home = "/home/dooncloud/桌面/practice/8.5face_reg_three_emotion/train_pic/home_of_allpics/"
''''''
def sum_all_the_pic(which_one):                                            # 图片是从边缘提取之后筛选出来的
    path_file_number = glob.glob(Path_of_the_store_of_allimagesfloders+which_one+'/*')
    return len(path_file_number)

def save_pics(sum,which_one,start_numb_b):
        i = 0
        for name in glob.glob(Path_of_the_store_of_allimagesfloders+which_one+'/*'):
                i +=1 #first pic
                pic = cv2.imread(name)
                cv2.imwrite(put_allpics_into_home+str(start_numb_b)+'.jpg',pic)
                start_numb_b += 1 
    
def pic_deel(label):
        para_init()
        output_feedback = []
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
                output_feedback.append(start_numb)
                #print(output_feedback[which_one])
                j += label_sum[which_one]
        return output_feedback


# if __name__ == "__main__":
    
#     main()
# print('将上面区间填入2-face_datasets.py _getitem_中即可')