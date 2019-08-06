import cv2
import glob
import numpy
from PIL import Image
import os
'''
将筛选过后的文件夹中图片按照顺序重新命名
注意已经修改过序号的图片混合未命名图片的时候 名字要加上字母寓意区别
'''

folder_path = 'normal/'  # 改这两处
# cv2.imwrite("/home/dooncloud/桌面/practice/8.5face_reg_three_emoti 
label = ["smile",'hand_on_face','half_face','normal'] # 更改标签
label_sum = []#存放图片


images = "/home/dooncloud/桌面/practice/8.5face_reg_three_emotion/train_pic/"+folder_path
def sum_all_the_pic():                                            # 图片是从边缘提取之后筛选出来的
    path_file_number = glob.glob(pathname='/home/dooncloud/桌面/practice/8.5face_reg_three_emotion/train_pic/'+folder_path+'*.jpg')
    #print(len(path_file_number))
    return len(path_file_number)

def save_pics(sum):
    i = 0
    j = 1
    for name in glob.glob('/home/dooncloud/桌面/practice/8.5face_reg_three_emotion/train_pic/'+folder_path+'*'):
        i +=1 #first pic
        try:
            pic = cv2.imread(name)
            cv2.imwrite("/home/dooncloud/桌面/practice/8.5face_reg_three_emotion/train_pic/"+folder_path+str(i)+'.jpg',pic)
            j += 1 
            os.remove(name)
            #print(i)
                
        except FileNotFoundError:
            continue
        except TypeError:
            continue
        except OSError:
            continue
    # print(j)

def main():
    # for name in glob.glob('/home/dooncloud/桌面/practice/8.5face_reg_three_emotion/train_pic/half_face/*'):
    



if __name__ == "__main__":
    sum_pics = sum_all_the_pic()
    print(sum_pics)
    save_pics(sum_pics)
    #main()