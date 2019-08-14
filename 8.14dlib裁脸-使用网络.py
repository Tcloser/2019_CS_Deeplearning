# -*- coding:utf-8 -*-
import face_recognition as fr
from PIL import Image, ImageDraw
import os,json,cv2,time
import numpy as np
import glob

filepath=glob.glob("/home/ubuntu/MyTestFile/Pic_Downloader/image/*")
print(filepath)
k = 1
yaq = 0
ldq = 0
for yaq,i in enumerate(filepath):
    if yaq < 0:continue
    print(str(yaq+1) + "/" + str(len(filepath)))
    print (i)
    try:
        image = cv2.imread(i)
        #print image.shape
        stu_img = fr.load_image_file(os.path.join(i))  
        # stu_img = fr.load_image_file(i)  
    except IOError:
        # os.remove(i)
        print ('del the bad pic' + i)
        continue
    #stu_img = cv2.imread(i)
    try:
        zzz = image.shape
    except AttributeError:
        continue
    if image.shape[0] > 1500 or image.shape[1] > 1500:
        if image.shape[0] > image.shape[1]:
            size = (1500,int(1500*(image.shape[0]/(image.shape[0]*1.))))
        else:
            size = (int(1500*(image.shape[0]/(image.shape[0]*1.))),1500)
        image = cv2.resize(image, size)
        stu_img = cv2.resize(stu_img, size)
        zzz = stu_img.shape
    try:
        # stu_loc = fr.face_locations(stu_img, model="cnn")
        facelocation_ = fr.face_locations(stu_img, number_of_times_to_upsample=2 ,model="cnn")
        face_landmarks_list = fr.face_landmarks(stu_img,face_locations=facelocation_, model="large")
    except RuntimeError:
        print ('error,rerun from ' + str(ldq + 1))
        break
        # f_1 = open('where.txt','r+')
        # f_1.write(str(i))
        # f_1.close()
        continue
    ldq = yaq
    # print(face_landmarks_list)
    if len(face_landmarks_list) >= 1:
        # for l in stu_loc:
        # face_landmarks_list = face_recognition.face_landmarks(image, model='large')
        # print("I found {} face(s) in this photograph.".format(len(face_landmarks_list)))
        # # 输出第一个人脸的面部特征信息
        # print(face_landmarks_list[0]["chin"])
        # print(face_landmarks_list[0]["left_eyebrow"])
        # print(face_landmarks_list[0]["right_eyebrow"])
        # print(face_landmarks_list[0]["nose_bridge"])
        # print(face_landmarks_list[0]["nose_tip"])
        # print(face_landmarks_list[0]["left_eye"])
        # print(face_landmarks_list[0]["right_eye"])
        # print(face_landmarks_list[0]["top_lip"])
        # print(face_landmarks_list[0]["bottom_lip"])
        pil_image = Image.fromarray(stu_img)
        d = ImageDraw.Draw(pil_image)
        for face_landmarks in face_landmarks_list:
            facial_features = [
            'chin',
            'left_eyebrow',
            'right_eyebrow',
            'nose_bridge',
            'nose_tip',
            'left_eye',
            'right_eye',
            'top_lip',
            'bottom_lip'
            ]
            for facial_feature in facial_features:
                # if(facial_feature == 'nose_bridge'):
                #     print '*************'
                #     print face_landmarks[facial_feature]
                #     print '*************'
                d.line(face_landmarks[facial_feature], width=1)
                # d.point(face_landmarks[facial_feature][8]) #chin
                # d.point(face_landmarks[facial_feature][0]) #left_eye
                # d.point(face_landmarks[facial_feature][3]) #right_eye
                # d.point(face_landmarks[facial_feature][2]) #nose_tip
                # d.point(face_landmarks[facial_feature][3]) #nose_bridge
            #     pass
            # pass
            dooncloud_line = [[0,0],[0,0]]
            print (face_landmarks['left_eye'])
            left_eye = list(face_landmarks['left_eye'][0])
            # face_landmarks = list(face_landmarks)
            right_eye = list(face_landmarks['right_eye'][3])
            chin = list(face_landmarks['chin'][8])
            print (left_eye[0] + right_eye[0])
            # dooncloud_line[0] = (((face_landmarks['left_eye'][0] + face_landmarks['right_eye'][3]) / 2.) + face_landmarks['chin'][8]) / 2.
            dooncloud_line[0][0] = (((left_eye[0] + right_eye[0]) / 2. + chin[0] ) / 2.) + ( (left_eye[0] + right_eye[0]) / 2. - chin[0] ) * 0.22
            dooncloud_line[0][1]=(((left_eye[1] + right_eye[1]) / 2. + chin[1] ) / 2.) + ( (left_eye[1] + right_eye[1]) / 2. - chin[1] ) * 0.22
            d_ = list(face_landmarks['nose_bridge'][3])
            dooncloud_line[1][0] = ( (d_[0]) + (d_[0] - dooncloud_line[0][0])* 150 )
            dooncloud_line[1][1] = ( (d_[1]) + (d_[1] - dooncloud_line[0][1])* 150 )
            dooncloud_line[0] = tuple(dooncloud_line[0])
            dooncloud_line[1] = tuple(dooncloud_line[1])
            print (tuple(dooncloud_line))
            d.line(tuple(dooncloud_line),width=5,fill=(255,0,0))
            d.line( [(0,0),(100,100)] ,width=10,fill=(0,255,255))
        del d
        # pil_image.show()
        pil_image.save("/home/ubuntu/MyTestFile/Pic_Downloader/img2/tn0_"+str(k)+'.jpg')
        k = k + 1
        pass
            # # cv2.rectangle(image, (l[3],l[0]), (l[1],l[2]), (0,255,0), 2)
            # h = (l[2] - l[0])
            # w = (l[1] - l[3])
            # print(l[0],l[2],l[3],l[1])
            # # cropped = image[l[0]-h/2:l[2]+2*h,l[3]-w:l[1]+w]
            # cropped = image[l[0]-h/2:l[2]+h/1,l[3]-w/1:l[1]+w/1]
            # # if cropped.shape[0] > 5 * cropped.shape[1] or cropped.shape[1] > 5 * cropped.shape[0]:continue
            # cv2.imwrite("/home/ubuntu/MyTestFile/Pic_Downloader/img/tn0_"+str(k)+'.jpg', cropped)
    #         k = k + 1
    #         print '==>' + str(k+1)
    # print '==>' + str(k+1)

        # cv2.imwrite("/home/ubuntu/MyTestFile/Pic_Downloader/img/"+str(k)+'.jpg', image)
    # else:
    #     # cv2.imwrite("/home/ubuntu/MyTestFile/Pic_Downloader/img/"+str(k)+'.jpg', image)
    #     continue
    # k = k + 1
    # cv2.imshow("1", stu_img)
    # cv2.waitKey(0)
# f_1 = open('where.txt','r+')
# f_1.write(str(0))
# f_1.close()
