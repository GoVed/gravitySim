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
from dataclasses import dataclass

#constants
g=6.674e-11

class PyData:
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
    
    #radius
    r=[]
    
    #reset to null state
    def reset(self):
        self.x.clear()
        self.y.clear()
        self.vx.clear()
        self.vy.clear()
        self.m.clear()
        self.r.clear()

class NpData:
    x=None
    y=None
    x2=None
    y2=None
    vx=None
    vy=None
    m=None
    r=None
    
    #Sync with pyData
    def __init__(self,pyData:PyData):
        self.x=np.array(pyData.x,np.float64)
        self.x=np.repeat([self.x],len(pyData.x),0)
        self.y=np.array(pyData.y,np.float64)
        self.y=np.repeat([self.y],len(pyData.y),0)
        self.m=np.array(pyData.m,np.float64)
        self.m=np.repeat([self.m],len(pyData.x),0)
        
        self.x2=self.x.transpose()
        self.y2=self.y.transpose()
        self.vx=np.array(pyData.vx,np.float64)
        self.vy=np.array(pyData.vy,np.float64)        
        self.r=np.array(pyData.r,np.float64)
    
    #reset to null state
    def reset(self):
        self.x=None
        self.y=None
        self.vx=None
        self.vy=None
        self.m=None
        self.r=None

class CUDAData:
    pass

@dataclass
class Modes:
    cpu=0
    gpu=1
    cpunumpy=2
    cpunumba=3
    cpujit=4
    cudajit=5
    length=6
    modes=('CPU','GPU','Numpy','Numba Parallel','JIT Parallel','JIT CUDA')
    

class Sim:
    
    #Basic python data
    pyData=PyData()
    
    #Numpy data, make sure to init
    npData=None
    
    
    
    def syncNumpy(self):
        self.npData=None
        self.npData=NpData(self.pyData)
    
    #Save the current state
    def save(self,name=''):
        if name=='':
            name=round(time.time())+'.txt'
        save={'x':self.pyData.x,'y':self.pyData.y,'vx':self.pyData.vx,'vy':self.pyData.vy,'m':self.pyData.m,'r':self.pyData.r}
        jsonstr=json.dumps(save)
        sf = open(name, "w")
        sf.write(jsonstr)
        sf.close()
        return name
    
    #Load to a state
    def load(self,name):
        self.pyData.reset()
        lf=open(name,"r")
        ld=json.loads(lf.read())
        lf.close()
        self.pyData.x.extend(ld['x'])
        self.pyData.y.extend(ld['y'])
        self.pyData.vx.extend(ld['vx'])
        self.pyData.vy.extend(ld['vy'])
        self.pyData.m.extend(ld['m'])
        self.pyData.r.extend(ld['r'])
    
    #Add new object to current state
    def addObject(self,init_x,init_y,init_vx,init_vy,init_m,init_r=1,updateTemp=True):        
        self.pyData.x.append(init_x)
        self.pyData.y.append(init_y)
        self.pyData.vx.append(init_vx)
        self.pyData.vy.append(init_vy)
        self.pyData.m.append(init_m)
        self.pyData.r.append(init_r)
        
    def addRandomObjects(self,count=10,xr=1000,yr=1000,vxr=10,vyr=10,mr=50,rr=4):
        for i in range(count):
            self.addObject(random.randint(-xr,xr),random.randint(-yr,yr),random.randint(-vxr,vxr),random.randint(-vyr,vyr),random.randint(0,mr),rr,False)
        
    #Main calculation function
    def calc(self,time_period=0.01,mode=Modes.cpu):    
        self.funcName[mode](self,time_period)
    
    #Calculating a step on CPU using python data structure
    def calcCPU(self,time_period):
        newx=self.pyData.x[:]
        newy=self.pyData.y[:]
        i=0
        while i<len(self.pyData.x):
            j=i+1
            while j<len(self.pyData.x):
                try:
                    
                    #calculating direction and force with each object
                    dir = math.atan2((self.pyData.y[i]-self.pyData.y[j]),(self.pyData.x[j]-self.pyData.x[i]))                
                    f=g*(self.pyData.m[i]*self.pyData.m[j])/(((self.pyData.y[j]-self.pyData.y[i])**2)+((self.pyData.x[j]-self.pyData.x[i])**2))  
                    
                    #adding accelaration to velocity
                    self.pyData.vx[i]+=f*math.cos(dir)/self.pyData.m[i]*time_period
                    self.pyData.vy[i]-=f*math.sin(dir)/self.pyData.m[i]*time_period 
                    self.pyData.vx[j]-=f*math.cos(dir)/self.pyData.m[j]*time_period
                    self.pyData.vy[j]+=f*math.sin(dir)/self.pyData.m[j]*time_period   
                except Exception as e:
                    print('Error occured while calculating, simulation might have huge errors!',e)
                    
                j+=1
            
            
            #adding displacement from velocity
            newx[i]+=self.pyData.vx[i]*time_period
            newy[i]+=self.pyData.vy[i]*time_period
            
            i+=1
    
        self.pyData.x=newx[:]
        self.pyData.y=newy[:]
        
    #Calculating a step on CPU using numpy data structure
    def calcNumpy(self,time_period):
        
        #Calculating accelaration
        pf=g*np.divide(self.npData.m,(np.add((np.power(np.subtract(self.npData.y2,self.npData.y),2)),np.power(np.subtract(self.npData.x2,self.npData.x),2))))
        
        #calculating direction
        dirc=np.arctan2(np.subtract(self.npData.y,self.npData.y2),np.subtract(self.npData.x,self.npData.x2))
        
        #getting x and y component of accelaration
        accx=np.multiply(pf,np.cos(dirc))
        accy=np.multiply(pf,np.sin(dirc)) 
                
        
        #Diagonal elements are calculation over same objects
        np.fill_diagonal(accx,0)
        np.fill_diagonal(accy,0)
        
        #Updating the velocity
        self.npData.vx+=np.sum(accx,axis=1)*time_period
        self.npData.vy+=np.sum(accy,axis=1)*time_period
        
        #Updating positions
        self.npData.x[0,:]+=self.npData.vx*time_period
        self.npData.y[0,:]+=self.npData.vy*time_period
        
        #Updating the structure with new position
        self.npData.x=np.repeat([self.npData.x[0,:]],len(self.pyData.x),0)
        self.npData.y=np.repeat([self.npData.y[0,:]],len(self.pyData.x),0)
        
    def calcGPU(self,time_period):
        #Calculate accelaration for each object on GPU
        
        accx=self.calcDirectAccXOnGPU(self.npData.x,self.npData.y,self.npData.x2,self.npData.y2,self.npData.m)
        accy=self.calcDirectAccYOnGPU(self.npData.x,self.npData.y,self.npData.x2,self.npData.y2,self.npData.m)
        
        #Diagonal elements are calculation over same objects
        np.fill_diagonal(accx,0)
        np.fill_diagonal(accy,0)
        
        # #Updating the velocity
        self.npData.vx+=np.sum(accx,axis=1)*time_period
        self.npData.vy+=np.sum(accy,axis=1)*time_period
        
        # #Updating positions
        # print('---------------')
        
        tx=np.copy(self.npData.x)
        ty=np.copy(self.npData.y)
        
        # px=np.copy(tx)        
        # py=np.copy(ty)
        
        tx[0,:]+=self.npData.vx*time_period
        ty[0,:]+=self.npData.vy*time_period
        
        #Updating the structure with new position
        tx=np.repeat([tx[0,:]],len(self.pyData.x),0)
        ty=np.repeat([ty[0,:]],len(self.pyData.x),0)
        
        # self.npData.x=tx
        # self.npData.y=ty
       
        # self.npData.x[0,:]+=self.npData.vx*time_period
        # self.npData.y[0,:]+=self.npData.vy*time_period
        
        # #Updating the structure with new position
        # self.npData.x=np.repeat([self.npData.x[0,:]],len(self.pyData.x),0)
        # self.npData.y=np.repeat([self.npData.y[0,:]],len(self.pyData.x),0)
        
        #update velocity and position of each object
        i=0
        while i<len(self.pyData.x):
            # self.npData.vx[i]+=np.sum(accx[i,:])*time_period
            # self.npData.vy[i]+=np.sum(accy[i,:])*time_period
            
            self.npData.x[:,i]+=self.npData.vx[i]*time_period
            self.npData.y[:,i]+=self.npData.vy[i]*time_period
            
            
            
            i+=1
        # print('-----------------------')
        # print("From method B",tx,ty)
        # print("From method A",self.npData.x,self.npData.y)
        if not np.allclose(tx,self.npData.x) or not np.allclose(ty,self.npData.y):
            
            print(tx,ty)
            print(self.npData.x,self.npData.y)
        else:
            self.npData.x=np.copy(tx)
            self.npData.y=np.copy(ty)
        
    #Numba GPU
    @vectorize(['float64(float64,float64,float64,float64,float64)'],target='cuda')
    def calcDirectAccXOnGPU(x1,y1,x2,y2,m2):   
        return g*m2/(((y2-y1)**2)+((x2-x1)**2))*math.cos(math.atan2((y1-y2),(x1-x2)))
    
    @vectorize(['float64(float64,float64,float64,float64,float64)'],target='cuda')
    def calcDirectAccYOnGPU(x1,y1,x2,y2,m2):   
        return g*m2/(((y2-y1)**2)+((x2-x1)**2))*math.sin(math.atan2((y1-y2),(x1-x2)))
        
        
    #for benchmarking / testing efficiency  
    def benchmark(self,objects=100,calcn=10000,frameWise=False,mode=Modes.cpu):
        print('Starting Benchmark')
        print('Resetting the environment')
        self.pyData.reset()
        print('Creating random objects')
        for i in range(objects):
            self.addObject(random.randint(-5000,5000),random.randint(-5000,5000),random.randint(-5000,5000),random.randint(-5000,5000),random.randint(0,1e18),5,False)
        
        if mode==2:
            print('Generating Numpy array')
            self.syncNumpy()
        
        print('starting calculations')
        start_time=time.time()
        for i in range(calcn):
            if calcn>100:
                if i%(round(calcn/100))==0:
                    print(round(i/calcn*100))
            else:
                print(i)
            self.calc(0.01,mode)
        end_time=time.time()
        print('Completed',calcn,'steps for',objects,'objects in',(end_time-start_time),'seconds')
        print('Calc/sec=',calcn/(end_time-start_time))
        
    #for calling out specific function
    funcName={0:calcCPU,1:calcGPU,2:calcNumpy}