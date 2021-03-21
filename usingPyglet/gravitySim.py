# -*- coding: utf-8 -*-
"""
Created on Sun Feb  7 16:08:39 2021

@author: vedhs
"""
import math
import random
import time
from numba import vectorize,jit,cuda
import numpy as np
import json
#constants
g=6.674e-11

#position
x=[]
y=[]

#velocity
vx=[]
vy=[]

#mass
m=[]
r=[]

#for calc
temp=[]

#reset to null state
def reset():
    x.clear()
    y.clear()
    vx.clear()
    vy.clear()
    m.clear()
    r.clear()

#Save the current state
def save(name=''):
    if name=='':
        name=round(time.time())+'.txt'
    save={'x':x,'y':y,'vx':vx,'vy':vy,'m':m,'r':r}
    jsonstr=json.dumps(save)
    sf = open(name, "w")
    sf.write(jsonstr)
    sf.close()
    return name
    

#Load to a state
def load(name):
    reset()
    lf=open(name,"r")
    ld=json.loads(lf.read())
    lf.close()
    x.extend(ld['x'])
    y.extend(ld['y'])
    vx.extend(ld['vx'])
    vy.extend(ld['vy'])
    m.extend(ld['m'])
    r.extend(ld['r'])
    genTemp()



#Add new object to current state
def addObject(init_x,init_y,init_vx,init_vy,init_m,init_r=1,updateTemp=True):
    
    x.append(init_x)
    y.append(init_y)
    vx.append(init_vx)
    vy.append(init_vy)
    m.append(init_m)
    r.append(init_r)
    if updateTemp:
        genTemp()
        
def addRandomObjects(count=10,xr=1000,yr=1000,vxr=10,vyr=10,mr=50,rr=4):
    for i in range(count):
        addObject(random.randint(-xr,xr),random.randint(-yr,yr),random.randint(-vxr,vxr),random.randint(-vyr,vyr),random.randint(0,mr),rr,False)
    genTemp()
    
#Function to generate cache, required before calculations
def genTemp():
    temp.clear()
    x1=np.array([x,]*len(x),dtype=np.float64)
    y1=np.array([y,]*len(x),dtype=np.float64)
    m1=np.array([m,]*len(x),dtype=np.float64)
    x2=x1.transpose()
    y2=y1.transpose()
    m2=m1.transpose()
    temp.append(x2) #0
    temp.append(y2) #1
    temp.append(m2) #2
    temp.append(x1) #3
    temp.append(y1) #4
    temp.append(m1) #5
    temp.append(np.array(vx,dtype=np.float64)) #6
    temp.append(np.array(vy,dtype=np.float64)) #7
    temp.append(cuda.to_device(temp[0])) #8
    temp.append(cuda.to_device(temp[1])) #9
    temp.append(cuda.to_device(temp[3])) #10
    temp.append(cuda.to_device(temp[4])) #11
    temp.append(cuda.to_device(temp[5])) #12       
    temp.append(cuda.to_device(temp[6])) #13
    temp.append(cuda.to_device(temp[7])) #14
    

#Main calculation function, supports GPU and CPU
def calc(time_period,sep=True,mode='cpu'):
    global x,y,vx,vy,m
    
    #Generate seperate array if setted to true
    newx=x
    newy=y
    if sep and mode != 'gpu' and mode!= 'numpycpu' and mode != 'numbacpu':
        newx=x[:]
        newy=y[:]
    if mode=='gpu':
        
        #Calculate accelaration for each object on GPU
        
        accx=calcDirectAccXOnGPU(temp[0],temp[1],temp[3],temp[4],temp[5])
        accy=calcDirectAccYOnGPU(temp[0],temp[1],temp[3],temp[4],temp[5])
        np.fill_diagonal(accx,0)
        np.fill_diagonal(accy,0)
        i=0
        
        #update velocity and position of each object
        while i<len(x):
            vx[i]-=np.sum(accx[i,:])*time_period
            vy[i]-=np.sum(accy[i,:])*time_period
            
            newx[i]+=vx[i]*time_period
            newy[i]+=vy[i]*time_period
            
            #updating temp array for calculation efficiency
            temp[0][:][i]=newx[i]
            temp[1][:][i]=newy[i]
            
            i+=1
        
        
    elif mode=='cpu':
        i=0
        while i<len(x):
            j=i+1
            while j<len(x):
                try:
                    
                    #calculating direction and force with each object
                    dir = math.atan2((y[i]-y[j]),(x[j]-x[i]))                
                    f=g*(m[i]*m[j])/(((y[j]-y[i])**2)+((x[j]-x[i])**2))  
                    
                    #adding accelaration to velocity
                    vx[i]+=f*math.cos(dir)/m[i]*time_period
                    vy[i]-=f*math.sin(dir)/m[i]*time_period 
                    vx[j]-=f*math.cos(dir)/m[j]*time_period
                    vy[j]+=f*math.sin(dir)/m[j]*time_period   
                except Exception as e:
                    print('Error occured while calculating, simulation might have huge errors!',e)
                    
                j+=1
            
            
            #adding displacement from velocity
            newx[i]+=vx[i]*time_period
            newy[i]+=vy[i]*time_period
            
            #updating temp array for calculation efficiency
            temp[0][:][i]=newx[i]
            temp[1][:][i]=newy[i]
            i+=1
            
    elif mode=='numpycpu':
        accx=calcDirectAccXOnNumpy(temp[0],temp[1],temp[3],temp[4],temp[5])
        accy=calcDirectAccYOnNumpy(temp[0],temp[1],temp[3],temp[4],temp[5])
        np.fill_diagonal(accx,0)
        np.fill_diagonal(accy,0)
        i=0
        
        #update velocity and position of each object
        while i<len(x):
            vx[i]-=np.sum(accx[i,:])*time_period
            vy[i]-=np.sum(accy[i,:])*time_period
            
            newx[i]+=vx[i]*time_period
            newy[i]+=vy[i]*time_period
            
            #updating temp array for calculation efficiency
            temp[0][:][i]=newx[i]
            temp[1][:][i]=newy[i]
            
            i+=1
            
    elif mode=='numbacpu':
        
        #Calculate accelaration for each object on CPU parallel
        
        accx=calcDirectAccXOnCPU(temp[0],temp[1],temp[3],temp[4],temp[5])
        accy=calcDirectAccYOnCPU(temp[0],temp[1],temp[3],temp[4],temp[5])
        np.fill_diagonal(accx,0)
        np.fill_diagonal(accy,0)
        i=0
        
        #update velocity and position of each object
        while i<len(x):
            vx[i]-=np.sum(accx[i,:])*time_period
            vy[i]-=np.sum(accy[i,:])*time_period
            
            newx[i]+=vx[i]*time_period
            newy[i]+=vy[i]*time_period
            
            #updating temp array for calculation efficiency
            temp[0][:][i]=newx[i]
            temp[1][:][i]=newy[i]
            
            i+=1
            
    if mode=='jit':
        
        #Calculate accelaration for each object using jit
        
        accx=calcDirectAccXOnJit(temp[0],temp[1],temp[3],temp[4],temp[5])
        accy=calcDirectAccYOnJit(temp[0],temp[1],temp[3],temp[4],temp[5])
        
        i=0
        
        #update velocity and position of each object
        while i<len(x):
            vx[i]-=accx[i]*time_period
            vy[i]-=accy[i]*time_period
            
            newx[i]+=vx[i]*time_period
            newy[i]+=vy[i]*time_period
            
            #updating temp array for calculation efficiency
            temp[0][:][i]=newx[i]
            temp[1][:][i]=newy[i]
            
            i+=1
            
    if mode=='cudajit':
        
        #Calculate accelaration for each object using cuda jit
        
        
        
        # Configure the blocks
        threadsperblock = (1,1)
        blockspergrid_x = int(math.ceil(temp[0].shape[0] / threadsperblock[0]))
        blockspergrid_y = int(math.ceil(temp[1].shape[1] / threadsperblock[1]))
        blockspergrid = (blockspergrid_x, blockspergrid_y)
        
        calcDirectAccOnCudaJit[blockspergrid, threadsperblock](temp[8],temp[9],temp[10],temp[11],temp[12],temp[13],temp[14],time_period)
        
        temp[13].copy_to_host(temp[6])
        temp[14].copy_to_host(temp[7])
        vx=temp[6].tolist()
        vy=temp[7].tolist()
        
        i=0
        
        #update velocity and position of each object
        while i<len(x):            
            
            newx[i]+=vx[i]*time_period
            newy[i]+=vy[i]*time_period
            
            #updating temp array for calculation efficiency
            temp[0][:][i]=newx[i]
            temp[1][:][i]=newy[i]
            
            i+=1
        temp[8]=cuda.to_device(temp[0])
        temp[9]=cuda.to_device(temp[1])
        temp[10]=cuda.to_device(temp[3]) 
        temp[11]=cuda.to_device(temp[4]) 
            
   
    #updating position if seperated
    if sep and mode != 'gpu' and mode!= 'numpycpu' and mode != 'numbacpu':
        x=newx[:]
        y=newy[:]
        # i=0
        # while i<len(x):
        #     x[i]=newx[i]
        #     y[i]=newy[i]        
        #     i+=1
#For numpy
def calcDirectAccXOnNumpy(x1,y1,x2,y2,m2):   
    return g*np.multiply(np.divide(m2,(np.add((np.power(np.subtract(y2,y1),2)),np.power(np.subtract(x2,x1),2)))),np.cos(np.arctan2(np.subtract(y1,y2),np.subtract(x1,x2))))


def calcDirectAccYOnNumpy(x1,y1,x2,y2,m2):   
    return g*np.multiply(np.divide(m2,(np.add((np.power(np.subtract(y2,y1),2)),np.power(np.subtract(x2,x1),2)))),np.sin(np.arctan2(np.subtract(y1,y2),np.subtract(x1,x2))))

#Using cuda.jit
@cuda.jit(inline=True)
def calcDirectAccOnCudaJit(x1,y1,x2,y2,m2,vx,vy,t):  
    i,j=cuda.grid(2)
    if i < x1.shape[0]:    
        if j < x1.shape[1]:
            if i!=j:
                vx[i]-=g*m2[i,j]/(((y2[i,j]-y1[i,j])**2)+((x2[i,j]-x1[i,j])**2))*math.cos(math.atan2((y1[i,j]-y2[i,j]),(x1[i,j]-x2[i,j])))*t
                vy[i]-=g*m2[i,j]/(((y2[i,j]-y1[i,j])**2)+((x2[i,j]-x1[i,j])**2))*math.sin(math.atan2((y1[i,j]-y2[i,j]),(x1[i,j]-x2[i,j])))*t
        
                
    

#Using jit
@jit(nopython=True, parallel=True)
def calcDirectAccXOnJit(x1,y1,x2,y2,m2):   
    result = np.zeros(len(x1))
    for i in range(x1.shape[0]):
        for j in range(x1.shape[1]):
            if i!=j:
                result[i]+=g*m2[i,j]/(((y2[i,j]-y1[i,j])**2)+((x2[i,j]-x1[i,j])**2))*math.cos(math.atan2((y1[i,j]-y2[i,j]),(x1[i,j]-x2[i,j])))
    return result

@jit(nopython=True, parallel=True)
def calcDirectAccYOnJit(x1,y1,x2,y2,m2):  
    result = np.zeros(len(x1))
    for i in range(x1.shape[0]):
        for j in range(x1.shape[1]):
            if i!=j:
                result[i]+=g*m2[i,j]/(((y2[i,j]-y1[i,j])**2)+((x2[i,j]-x1[i,j])**2))*math.sin(math.atan2((y1[i,j]-y2[i,j]),(x1[i,j]-x2[i,j])))
    return result 

#Numba GPU
@vectorize(['float64(float64,float64,float64,float64,float64)'],target='cuda')
def calcDirectAccXOnGPU(x1,y1,x2,y2,m2):   
    return g*m2/(((y2-y1)**2)+((x2-x1)**2))*math.cos(math.atan2((y1-y2),(x1-x2)))

@vectorize(['float64(float64,float64,float64,float64,float64)'],target='cuda')
def calcDirectAccYOnGPU(x1,y1,x2,y2,m2):   
    return g*m2/(((y2-y1)**2)+((x2-x1)**2))*math.sin(math.atan2((y1-y2),(x1-x2)))


#Numba multi-CPU
@vectorize(['float64(float64,float64,float64,float64,float64)'],target='parallel')
def calcDirectAccXOnCPU(x1,y1,x2,y2,m2):   
    return g*m2/(((y2-y1)**2)+((x2-x1)**2))*math.cos(math.atan2((y1-y2),(x1-x2)))

@vectorize(['float64(float64,float64,float64,float64,float64)'],target='parallel')
def calcDirectAccYOnCPU(x1,y1,x2,y2,m2):   
    return g*m2/(((y2-y1)**2)+((x2-x1)**2))*math.sin(math.atan2((y1-y2),(x1-x2)))



#calculate n steps     
def calcn(n=100,t=0.01,sep=True,mode='cpu'):
    for i in range(n):
        calc(t,sep,mode)



        
#for testing efficiency    
def testBenchmark(objects=100,calcn=10000,frameWise=False,mode='cpu'):
    print('Resetting the environment')
    reset()
    print('Creating random objects')
    for i in range(objects):
        addObject(random.randint(-5000,5000),random.randint(-5000,5000),random.randint(-5000,5000),random.randint(-5000,5000),random.randint(0,1e18),5,False)
    genTemp()
    print('starting calculations')
    start_time=time.time()
    for i in range(calcn):
        if calcn>100:
            if i%(round(calcn/100))==0:
                print(round(i/calcn*100))
        else:
            print(i)
        calc(0.01,False,mode)
    end_time=time.time()
    print('Completed',calcn,'steps for',objects,'objects in',(end_time-start_time),'seconds')
    print('Calc/sec=',calcn/(end_time-start_time))
    
    
    