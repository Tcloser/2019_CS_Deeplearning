1 训练集分类之后放到train_pic下 同时修改 train_predict.py中图片路径
2 更改 put_pics_into_home_andrename.py 参数初始化中的文件夹名称和存储路径  
分类数量不为4的时候
        if   index < pics_nub_list[0]:
            target = 0
        elif index < pics_nub_list[1]:
            target = 1
        elif index < pics_nub_list[2]:
            target = 2
        else:
            target = 3
        return image, target
修改一下
