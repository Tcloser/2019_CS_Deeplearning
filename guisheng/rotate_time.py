#############################################adam###############################
import torch, os, random, argparse
import torch.nn as nn
from torch.autograd import \
    Variable  
from torchvision import datasets, transforms
import time
import cv2
import numpy as np
from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt
import math
import pytorchtools #import EarlyStopping
from sympy import *  
from sympy.parsing.sympy_parser import parse_expr
from sympy import plot_implicit
torch.set_printoptions(threshold=1000000000)
np.set_printoptions(threshold=1000000000)
torch.manual_seed(777)
from mpl_toolkits.mplot3d import Axes3D

def cuda(*pargs):
    if torch.cuda.is_available():
        return (data.cuda() for data in pargs)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def ReadPQR(pqrfile):
    """
    get the minimum and maxinum of  x,y,z
    """
    pqr = open(pqrfile,'r')
    lines = pqr.readlines()
    coordinate = []
    r = []
    natom = 0
    for line in lines:
        if line[0:4] == "ATOM" or line[0:6] == "HETATM":
            natom += 1
            if len(line.strip().split()) == 10:
                record, serial, atomName, residueName, residueNum, x, y, z, charge, radius = line.strip().split()
            elif len(line.strip().split()) == 11:
                record, serial, atomName, residueName, chainID, residueNum, x, y, z, charge, radius = line.strip().split()
            else:

                x = line[30:38]
                y = line[38:46]
                z = line[46:54]

                # x = line[30:38]
                # y = line[38:46]
                # z = line[46:54]
            x = float(x)
            y = float(y)
            z = float(z)
            
            radius = float(radius)
            r.append(radius)
            coordinate.append([x,y,z])
    pqr.close()
    return coordinate, r

if __name__ == '__main__':
#################################################################################
###############################Read PQR FILE#####################################
#################################################################################
    # starttime = time.time()

    starttime = time.time()

    file = ReadPQR('/home/dooncloud/桌面/guisheng/1MAG.pqr')
    x0 = torch.tensor(file[0],dtype = torch.float)
    r = torch.tensor(file[1],dtype = torch.float)
    r = r.view(len(r),-1)

    endtime = time.time()
    print('ReadPQR:'+ str(endtime-starttime))

    gsrange = [torch.min(x0[:,0])-5,torch.max(x0[:,0])+5,torch.min(x0[:,1])-5,torch.max(x0[:,1])+5,torch.min(x0[:,2])-5,torch.max(x0[:,2])+5]
    ALPHA=1
    BETA=1


    starttime = time.time()

    batch_X = np.arange(gsrange[0],gsrange[1],1.5)
    batch_Y = np.arange(gsrange[2],gsrange[3],1.5)
    batch_Z = np.arange(gsrange[4],gsrange[5],1.5)

    batch_x,batch_y,batch_z = np.meshgrid(batch_X,batch_Y,batch_Z)
    batch_x = batch_x.flatten('F')
    batch_y = batch_y.flatten('F')
    batch_z = batch_z.flatten('F')
    batch_x = torch.tensor(batch_x,dtype = torch.float)
    batch_y = torch.tensor(batch_y,dtype = torch.float)
    batch_z = torch.tensor(batch_z,dtype = torch.float)

    batch_x = batch_x.view(batch_x.shape[0],-1)
    batch_y = batch_y.view(batch_y.shape[0],-1)
    batch_z = batch_z.view(batch_z.shape[0],-1)

    endtime = time.time()
    print('batch:'+ str(endtime-starttime))

    di = 0.5*torch.ones(x0.size(0),1)
    ci = torch.exp(di*r**2)
    ##########################get_label##############################
    starttime = time.time()

    zpp=0;
    for k in range(x0.size(0)):
        zpp=zpp+ci[k]*torch.exp(-di[k][0]*((batch_x-x0[k][0]).pow(2)+(batch_y-x0[k][1]).pow(2)+(batch_z-x0[k][2]).pow(2)))
    zp = zpp

    endtime = time.time()
    print('label:'+ str(endtime-starttime))


    starttime = time.time()

    mp = torch.sqrt(torch.max(zp))
    zp = torch.where(zp < 0.2, \
              torch.full_like(zp, 0), zp)

    #######idz = zp>3#########
    indices = (zp.squeeze(1) <= 3).nonzero().squeeze(1)
    zp = torch.index_select(zp,0,indices)
    batch_x = torch.index_select(batch_x,0,indices) 
    batch_y = torch.index_select(batch_y,0,indices) 
    batch_z = torch.index_select(batch_z,0,indices)
    
    gp=x0[:,0:3]
    Weight = nn.Parameter(torch.sqrt(ci))
    ddi_1 = torch.sqrt(di)
    ddi_2 = torch.sqrt(di)
    ddi_3 = torch.sqrt(di)
    Beta = nn.Parameter(torch.cat((ddi_1,ddi_2,ddi_3),1))
    Centers = nn.Parameter(gp)
    Angle = nn.Parameter(torch.zeros(x0.size(0),3,device='cuda'))
    gsrange = torch.tensor(gsrange)

    endtime = time.time()
    print('initialize:'+ str(endtime-starttime))


    starttime = time.time()
    Weight,Beta,Centers,Angle,zp,gsrange,batch_x,batch_y,batch_z = cuda(Weight,Beta,Centers,Angle,zp,gsrange,batch_x,batch_y,batch_z) 
    endtime = time.time()
    print('cuda:'+ str(endtime-starttime))

    MAX_ITER = 10001
    lr = 0.001
    eps = 1e-8 
    state_b = {}
    state_c = {}
    state_w = {}
    state_a = {}
    state_b['step'] = 0
    state_b['exp_avg'] = torch.zeros_like(Beta.data)
    state_b['exp_avg_sq'] = torch.zeros_like(Beta.data)
    state_c['step'] = 0
    state_c['exp_avg'] = torch.zeros_like(Centers.data)
    state_c['exp_avg_sq'] = torch.zeros_like(Centers.data)
    state_w['step'] = 0
    state_w['exp_avg'] = torch.zeros_like(Weight.data)
    state_w['exp_avg_sq'] = torch.zeros_like(Weight.data)   
    state_a['step'] = 0
    state_a['exp_avg'] = torch.zeros_like(Angle.data)
    state_a['exp_avg_sq'] = torch.zeros_like(Angle.data) 
    # print '\tniter \t\tmse \t\tw \t\tb \tcost'
    for epoch in range(0,MAX_ITER):

        starttime = time.time()    
        if epoch % 10 == 0:
            eps_1 = 1e-4
            indices = (abs(Weight.data.squeeze(1))>=eps_1).nonzero().squeeze(1)
            Weight.data = torch.index_select(Weight.data,0,indices)
            Beta.data = torch.index_select(Beta.data,0,indices)
            Centers.data = torch.index_select(Centers.data,0,indices) 
            Angle.data = torch.index_select(Angle.data,0,indices) 
            state_w['exp_avg'] = torch.index_select(state_w['exp_avg'],0,indices)
            state_w['exp_avg_sq'] = torch.index_select(state_w['exp_avg_sq'],0,indices)   
            state_b['exp_avg'] = torch.index_select(state_b['exp_avg'],0,indices)
            state_b['exp_avg_sq'] = torch.index_select(state_b['exp_avg_sq'],0,indices)
            state_c['exp_avg'] = torch.index_select(state_c['exp_avg'],0,indices)        
            state_c['exp_avg_sq'] = torch.index_select(state_c['exp_avg_sq'],0,indices)
            state_a['exp_avg'] = torch.index_select(state_a['exp_avg'],0,indices)        
            state_a['exp_avg_sq'] = torch.index_select(state_a['exp_avg_sq'],0,indices)
        Weight = nn.Parameter(Weight.data)
        Beta = nn.Parameter(Beta.data)
        Centers = nn.Parameter(Centers.data)
        Angle = nn.Parameter(Angle.data)

        endtime = time.time()
        print('delete:'+ str(endtime-starttime))


        starttime = time.time()
        A_x = batch_x
        A_y = batch_y
        A_z = batch_z
        B_x = Centers[:,0].repeat(A_x.size(0),1)
        B_y = Centers[:,1].repeat(A_y.size(0),1)
        B_z = Centers[:,2].repeat(A_z.size(0),1)

        th1 = Angle[:,0]
        th2 = Angle[:,1]
        th3 = Angle[:,2]
        star_1 = torch.cos(th2)*torch.cos(th3)
        star_2 = torch.cos(th2)*torch.sin(th3)
        star_3 = -torch.sin(th2)
        star_4 = torch.sin(th1)*torch.sin(th2)*torch.cos(th3) - torch.cos(th1)*torch.sin(th3)
        star_5 = torch.cos(th1)*torch.cos(th3) + torch.sin(th1)*torch.sin(th2)*torch.sin(th3)
        star_6 = torch.sin(th1)*torch.cos(th2)
        star_7 = torch.sin(th1)*torch.sin(th3) + torch.cos(th1)*torch.sin(th2)*torch.cos(th3)
        star_8 = torch.cos(th1)*torch.sin(th2)*torch.sin(th3) - torch.sin(th1)*torch.cos(th3)
        star_9 = torch.cos(th1)*torch.cos(th2)

        A_x = batch_x
        A_y = batch_y
        A_z = batch_z
        B_x = Centers[:,0].repeat(A_x.size(0),1)
        B_y = Centers[:,1].repeat(A_y.size(0),1)
        B_z = Centers[:,2].repeat(A_z.size(0),1)

        A_1 = (Beta[:,0].pow(2)*(star_1.pow(2)) + Beta[:,1].pow(2)*(star_4.pow(2)) + Beta[:,2].pow(2)*(star_7.pow(2))).mul((A_x - B_x).pow(2))
        A_2 = (Beta[:,0].pow(2)*(star_2.pow(2)) + Beta[:,1].pow(2)*(star_5.pow(2)) + Beta[:,2].pow(2)*(star_8.pow(2))).mul((A_y - B_y).pow(2))
        A_3 = (Beta[:,0].pow(2)*(star_3.pow(2)) + Beta[:,1].pow(2)*(star_6.pow(2)) + Beta[:,2].pow(2)*(star_9.pow(2))).mul((A_z - B_z).pow(2))
        A_4 = (2*(Beta[:,0].pow(2)*star_1*star_2 + Beta[:,1].pow(2)*star_4*star_5 + Beta[:,2].pow(2)*star_7*star_8)).mul((A_x - B_x)*(A_y - B_y))
        A_5 = (2*(Beta[:,0].pow(2)*star_1*star_3 + Beta[:,1].pow(2)*star_4*star_6 + Beta[:,2].pow(2)*star_7*star_9)).mul((A_x - B_x)*(A_z - B_z))
        A_6 = (2*(Beta[:,0].pow(2)*star_2*star_3 + Beta[:,1].pow(2)*star_5*star_6 + Beta[:,2].pow(2)*star_8*star_9)).mul((A_y - B_y)*(A_z - B_z))
        C = torch.exp(-A_1 - A_2 - A_3 - A_4 - A_5 - A_6) 
        W_pow = Weight.pow(2)
        zg = C.mm(W_pow)

        endtime = time.time()
        print('y_prediction:'+ str(endtime-starttime))





        starttime = time.time()
        mse = (((zg - zp).pow(2)).sum())
        w = torch.norm(Weight,1)
        b = torch.norm(Beta[:,0],1) + torch.norm(Beta[:,1],1) + torch.norm(Beta[:,2],1)
        cost = ALPHA*mse + BETA*(w + b)
        endtime = time.time()
        print('cost:'+ str(endtime-starttime))


        starttime = time.time()
        cost.backward()
        endtime = time.time()
        print('backward:'+ str(endtime-starttime))


        # print('M:{},W:{},B:{},C:{}'.format(mse,w,b,cost))

        # if epoch % 1 == 0:
        #     print '\t', epoch+1,'\t{} \t{} \t{} \t{} \t{}'.format(mse,w,b,cost,Weight.size())


        starttime = time.time()
        ########################optimization_Beta####################
        exp_avg, exp_avg_sq = state_b['exp_avg'], state_b['exp_avg_sq']
        state_b['step'] += 1
        beta1 = 0.9
        beta2 = 0.999
        exp_avg.mul_(beta1).add_(1 - beta1, Beta.grad.data)
        exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, Beta.grad.data, Beta.grad.data)             
        denom = exp_avg_sq.sqrt().add_(eps)
        bias_correction1 = 1 - beta1 ** state_b['step']
        bias_correction2 = 1 - beta2 ** state_b['step']
        step_size = lr * math.sqrt(bias_correction2) / bias_correction1
        Beta.data.addcdiv_(-step_size, exp_avg, denom)

        #######################optimization_Centers###################
        exp_avg, exp_avg_sq = state_c['exp_avg'], state_c['exp_avg_sq']
        state_c['step'] += 1
        beta1 = 0.9
        beta2 = 0.999
        exp_avg.mul_(beta1).add_(1 - beta1, Centers.grad.data)
        exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, Centers.grad.data, Centers.grad.data)             
        denom = exp_avg_sq.sqrt().add_(eps)
        bias_correction1 = 1 - beta1 ** state_c['step']
        bias_correction2 = 1 - beta2 ** state_c['step']
        step_size = lr * math.sqrt(bias_correction2) / bias_correction1
        Centers.data.addcdiv_(-step_size, exp_avg, denom)
        
        #######################optimization_Weight#####################
        exp_avg, exp_avg_sq = state_w['exp_avg'], state_w['exp_avg_sq']
        state_w['step'] += 1
        beta1 = 0.9
        beta2 = 0.999
        exp_avg.mul_(beta1).add_(1 - beta1, Weight.grad.data)
        exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, Weight.grad.data, Weight.grad.data)             
        denom = exp_avg_sq.sqrt().add_(eps)
        bias_correction1 = 1 - beta1 ** state_w['step']
        bias_correction2 = 1 - beta2 ** state_w['step']
        step_size = lr * math.sqrt(bias_correction2) / bias_correction1
        Weight.data.addcdiv_(-step_size, exp_avg, denom) 
        ######################optimization__Angle######################
        exp_avg, exp_avg_sq = state_a['exp_avg'], state_a['exp_avg_sq']
        state_a['step'] += 1
        beta1 = 0.9
        beta2 = 0.999
        exp_avg.mul_(beta1).add_(1 - beta1, Angle.grad.data)
        exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, Angle.grad.data, Angle.grad.data)             
        denom = exp_avg_sq.sqrt().add_(eps)
        bias_correction1 = 1 - beta1 ** state_a['step']
        bias_correction2 = 1 - beta2 ** state_a['step']
        step_size = lr * math.sqrt(bias_correction2) / bias_correction1
        Angle.data.addcdiv_(-step_size, exp_avg, denom) 

        endtime = time.time()
        print('optimization:'+ str(endtime-starttime))
        ######################Check center coordinate, weight, Beta ################################
        starttime = time.time()
        ########Beta###########       
        Beta.data = torch.where(abs(Beta.data) > 0.8, \
                  torch.full_like(Beta.data, 0.8), Beta.data) 
        #########x#############
        Centers[:,0] = torch.where(Centers[:,0] < gsrange[0], \
              torch.full_like(Centers[:,0], gsrange[0]),Centers[:,0])
        Centers[:,0] = torch.where(Centers[:,0] > gsrange[1], \
              torch.full_like(Centers[:,0], gsrange[1]),Centers[:,0])
        ##########y#########
        Centers[:,1] = torch.where(Centers[:,1] < gsrange[2], \
              torch.full_like(Centers[:,1], gsrange[2]),Centers[:,1])
        Centers[:,1] = torch.where(Centers[:,1] > gsrange[3], \
              torch.full_like(Centers[:,1], gsrange[3]),Centers[:,1])
        #########z##########
        Centers[:,2] = torch.where(Centers[:,2] < gsrange[4], \
              torch.full_like(Centers[:,2], gsrange[4]),Centers[:,2])
        Centers[:,2] = torch.where(Centers[:,2] > gsrange[5], \
              torch.full_like(Centers[:,2], gsrange[5]),Centers[:,2])
        ########Weight###########
        try:
            Weight.data = torch.where(abs(Weight.data) > mp, \
                      torch.full_like(Weight.data, mp), Weight.data)
        except RuntimeError:
            continue
        endtime = time.time()
        print('constraint:'+ str(endtime-starttime))



    starttime = time.time()

    print(Weight.shape)
    eps_2 = 1e-4
    eps_3 = 1e-3
    beta = (abs(Beta.data)<eps_3).sum(0)
    weight = (abs(Weight.data)<eps_2).sum(0)
    jiao = beta + weight 
    beta = (beta == 2).sum()
    weight = (weight == 1).sum()
    jiao = (jiao == 3).sum()
    print(jiao)
    print(beta - jiao)
    print(weight - jiao)

    endtime = time.time()
    print('stastics:'+ str(endtime-starttime))


    #nohup python3 rotate_time.py>1MAG_time.txt 2>&1&

    # endtime = time.time()
    # print('time:'+ str(endtime-starttime))
