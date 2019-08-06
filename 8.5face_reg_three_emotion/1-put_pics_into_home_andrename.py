import cv2
import glob
import numpy
from PIL import Image
import os
'''将筛选过后的文件夹中图片按照顺序重新命名存放到home_pic'''

label = ["smile",'hand_on_face','half_face','normal'] # 更改标签即可将这几个文件夹中的图片放入home_of_allpics
label_sum = []#存放图片数量


def sum_all_the_pic(which_one):                                            # 图片是从边缘提取之后筛选出来的
    path_file_number = glob.glob('/home/dooncloud/桌面/practice/8.5face_reg_three_emotion/train_pic/'+which_one+'/*')
    #print(len(path_file_number))
    return len(path_file_number)

def save_pics(sum,which_one,start_numb):
    i = 0

    for name in glob.glob('/home/dooncloud/桌面/practice/8.5face_reg_three_emotion/train_pic/'+which_one+'/*'):
        i +=1 #first pic
        try:
            pic = cv2.imread(name)
            #print(numpy.size(pic))
            if numpy.size(pic) != 1:
                cv2.imwrite("/home/dooncloud/桌面/practice/8.5face_reg_three_emotion/train_pic/home_of_allpics/"+str(start_numb)+'.jpg',pic)
                start_numb += 1 

        except FileNotFoundError:
            continue
        except TypeError:
            continue
        except OSError:
            continue
    
def main():
    # for name in glob.glob('/home/dooncloud/桌面/practice/8.5face_reg_three_emotion/train_pic/half_face/*'):
    start_numb = 1
    j = 1
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