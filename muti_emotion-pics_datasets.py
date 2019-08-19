import os,json,cv2,time
import numpy as np
import glob

# filepath=glob.glob("/home/dooncloud/桌面/正面微笑|understand/*")

k = 10000#换个名字开始保存

for name in glob.glob("/home/dooncloud/桌面/手托腮+鼻子|thinking/*"):
    print(name)

    image = cv2.imread(name)
    stu_img = image
    h_flip = cv2.flip(stu_img,-1)# 1-水平  0-垂直  -1-水平垂直



    cv2.imwrite("/home/dooncloud/桌面/1/"+str(k)+'2'+'.jpg', h_flip)
    k += 1