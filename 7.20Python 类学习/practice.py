class rec(object):
    '''小的注意点'''
    def __init__(self,name,place,favorate='lol'): #init 注意是双下划线
        self.name = name
        self.place = place
        self.favorate = favorate
        print(self.name+'res第一次调用已经结束了') #属性可以直接打印出来

    def xu(self,age1,job1,**kw): #res_son init无法继承age1和job属性
        self.age1 = age1
        self._job1 = job1
        #print(self.age1,'就当了',self._job1) 

    def watch_movie(self):
        if self.age1<18:
            print('%s 才 %s 只能看熊出没' %(self.name,self.age1))
        else:
            print('%s 可以看电影了'%self.name)

class rec_son(rec):# 继承 的2
    def __init__(self,name,place,favorate='joj'):
        super(rec_son,self).__init__(name,place,favorate)#继承rec中三个属性 在res中初始化了 这里不用写也能创建


    def playing(self):
        return self.name



class rec2(object):
    pass


def randm_para_add(*args):
    '''可变参数传入求和'''
    sum = 0
    for i in args:
        sum += i
    return sum

def max_gongyueshu(x,y):
    ''''''
    (x,y) = (y,x)if x>y else (x,y)
    for i in range(x,0,-1):
        if x%i ==0 and y%i ==0:
            return i
            break
        else:
            return 1

def global_para():
    '''变量作用域的试探'''
    # global a
    a = 100#虽然名字一样 但在俩个空间
    print(a)


def main_res_jicheng_test():
    recall_1 = rec('xu', 'beijing')
    recall_1.xu(17, 'teacher',transport='car')
    recall_son = rec_son('luodong', 'nanjing')#继承的时候先1、run res的init 然后2、run res_son的init

    recall_son.playing()
    # print(randm_para_add(1,2,3,4,5))

    # print(max_gongyueshu(11,6))

    # a = 200
    # global_para()
    # print(a)

def main_res_test():
    recall_1 = rec('xu', 'beijing')
    recall_1.xu(17, 'teacher',transport='car')
    recall_1.watch_movie()

    recall_2 = rec2()
    recall_2.name = 'tom'#可直接添加属性
    print(recall_2.name)


def main():
    pass

if __name__ == '__main__':

    main_res_jicheng_test()