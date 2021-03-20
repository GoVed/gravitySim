# -*- coding: utf-8 -*-
"""
Created on Sun Feb  7 12:52:56 2021

@author: vedhs
"""

import pygame as pg
import multiprocessing
import time
import math

#Initializing
pg.init()
#s_w=GetSystemMetrics(0)
#s_h=GetSystemMetrics(1)
s_w=1920
s_h=1080
disp=pg.display.set_mode((0,0),pg.FULLSCREEN)

#getting images
blackhole=pg.image.load('blackBody.png')
neutronStar=pg.image.load('whiteBody.png')
blueStar=pg.image.load('blueBody.png')
orangeStar=pg.image.load('orangeBody.png')
yellowStar=pg.image.load('yellowBody.png')
rockyBody=pg.image.load('brownBody.png')
trail=pg.transform.scale(neutronStar,(1,1))

#for parallel processing
pool = multiprocessing.Pool(4)


def calcFrame(calcx,calcy,vx,vy,m):
    try:
        j=0
        while j < len(x):
            if i ==j:
                j+=1
                continue
            if calcx[j] != calcx[i]:
                dir = math.degrees(math.atan((calcy[i]-calcy[j])/(calcx[j]-calcx[i])))
                if calcx[j] - calcx[i] < 0:
                    dir+=180
            else:
                if calcy[j] < calcy[i]:
                    dir = 0
                else:
                    dir=180
            
            f=G*(m[i]*m[j])/(((calcy[j]-calcy[i])**2)+((calcx[j]-calcx[i])**2))            
            vx[i]+=f*math.cos(math.radians(dir))/(m[i]*fps)
            vy[i]-=f*math.sin(math.radians(dir))/(m[i]*fps)
            j+=1
    except:
        print('Error occured')

#draw image
def drawBody(x,y,istrail=0,scale=1.0,body_type=5):
    x+=cx
    y+=cy
    if istrail==1:
        disp.blit(trail,(x,y))
    else:
        if body_type == 0:
            disp.blit(pg.transform.scale(blackhole,(int(100/scale),int(100/scale))),(x,y))
        elif body_type == 1:
            disp.blit(pg.transform.scale(neutronStar,(int(100/scale),int(100/scale))),(x,y))
        elif body_type == 2:
            disp.blit(pg.transform.scale(blueStar,(int(100/scale),int(100/scale))),(x,y))
        elif body_type == 3:
            disp.blit(pg.transform.scale(orangeStar,(int(100/scale),int(100/scale))),(x,y))
        elif body_type == 4:
            disp.blit(pg.transform.scale(yellowStar,(int(100/scale),int(100/scale))),(x,y))
        elif body_type == 5:
            disp.blit(pg.transform.scale(rockyBody,(int(100/scale),int(100/scale))),(x,y))
        
            

#for writing text
text = pg.font.SysFont('Comic Sans MS',15)
bigtext = pg.font.SysFont('Comic Sans MS',30)

#empty list for storage of pos and vel
x=[]
y=[]
vx=[]
vy=[]
m=[]
tx=[]
ty=[]
s=[]
bt=[]

#other vars
G=6.6674E-11
c=3.0E8
set_mass=False
md=0
sd=0
temp_scale=1.0
show_trails=True
body_type=5

#camera
cx=0
cy=0
isup=False
isdown=False
isleft=False
isright=False

#for getting fps
frames=0
fps=120
count_one_s=time.time()

#for getting vel on drag
tempx=-1
tempy=-1

quit=False
while not quit:
    #clearing screen
    disp.fill((10,10,10))
    #getting inputs
    for event in pg.event.get():
        
        if event.type == pg.QUIT:                
            quit=True
        if event.type == pg.KEYDOWN:
            #checking quit
            if event.key == pg.K_q:
                quit = True
            #reset logic
            if event.key == pg.K_r:
                x=[]
                y=[]
                vx=[]
                vy=[]
                m=[]
                tx=[]
                ty=[]
                s=[]
                bt=[]
                temp_scale=1.0
                tempx=-1
                tempy=-1
            #camera movement stuffs
            if event.key == pg.K_UP:
                isup=True
            if event.key == pg.K_DOWN:
                isdown=True
            if event.key == pg.K_LEFT:
                isleft=True
            if event.key == pg.K_RIGHT:
                isright=True
            #checking t pressed for trails
            if event.key == pg.K_t:
                if show_trails==True:
                    show_trails=False
                else:
                    show_trails=True
            #changing body type
            if event.key == pg.K_z:
                body_type -=1
                if body_type <0:
                    body_type=5
            if event.key == pg.K_x:
                body_type +=1
                if body_type > 5:
                    body_type=0
        if event.type == pg.KEYUP:
            #camera stuff
            if event.key == pg.K_UP:
                isup=False
            if event.key == pg.K_DOWN:
                isdown=False
            if event.key == pg.K_LEFT:
                isleft=False
            if event.key == pg.K_RIGHT:
                isright=False
        if event.type == pg.MOUSEBUTTONDOWN:  
            #putting new body
            if set_mass==False:                
                tempx=event.pos[0]-cx
                tempy=event.pos[1]-cy   
                                
        if event.type == pg.MOUSEBUTTONUP:   
            #logic of putting new body
            if set_mass==False:
                vx.append(event.pos[0]-cx-tempx)
                vy.append(event.pos[1]-cy-tempy)
                set_mass=True            
            else:
                m.append(10**(md/5))
                s.append(temp_scale)
                x.append(tempx)
                y.append(tempy)
                tempx=-1
                tempy=-1
                bt.append(body_type)
                set_mass=False
                
        if event.type == pg.MOUSEMOTION:
            #for live visualization of mass and size while putting body
            if set_mass == True:
                md=((s_h-event.pos[1])/s_h)*100 
                sd=((s_w-event.pos[0])/s_w)*100
                
                
    #hovering camera
    if isup:
        cy+=1
    if isdown:
        cy-=1
    if isright:
        cx-=1
    if isleft:
        cx+=1          
    
    
    
    #showing mass text
    if set_mass == True:
        if md != 0:
            show_smass = bigtext.render('Mass: '+str(10**(md/5))+' kg',False,(200,200,150))            
            disp.blit(show_smass,((s_w/2)-150,100))
            
    #Making trails
    while len(tx) > 500*len(x):
        tx.pop(0)
        ty.pop(0)
    i=0
    if show_trails:
        while i < len(tx):
            drawBody(tx[i],ty[i],istrail=1)
            i+=1
        
    #showing live body while making
    i=0    
    if tempx!=-1:
        if set_mass==False:
            drawBody(tempx-50,tempy-50)  
        else:
            temp_scale=math.sqrt(2)**((sd-50)/10)
            drawBody(tempx-(50/temp_scale),tempy-(50/temp_scale),scale=temp_scale,body_type=body_type)
            
    #constant coords for physics loading accuracy
    calcx=x[:]
    calcy=y[:]
    while i < len(x):
        #drawing bodies
        drawBody(x[i]-(50/s[i]),y[i]-(50/s[i]),scale=s[i],body_type=bt[i])
        show_mass = text.render(str(m[i]),False,(150,150,150))
        disp.blit(show_mass,(x[i]+cx,y[i]+cy))
        
        #physics
        pool.map(calcFrame, (calcx, calcy, vx, vy, m))
        
        x[i]+=vx[i]/fps
        y[i]+=vy[i]/fps
        tx.append(x[i])
        ty.append(y[i])        
        i+=1
        
    
    #counting and displaying fps
    frames += 1
    if time.time() - count_one_s >=1:
        fps=frames
        frames=0
        count_one_s=time.time()
    
    show_fps = text.render('PPS :'+str(fps),False,(255,255,255))
    disp.blit(show_fps,(s_w-500,20))
    
    
    #updating display
    pg.display.update()
        
pg.quit()
