# -*- coding: utf-8 -*-
"""
Created on Sun Feb  7 14:30:37 2021

@author: vedhs
"""

import pyglet
import gravitySim

gs=gravitySim.Sim()

#interface variables
width=1280
height=720

#generating window
win = pyglet.window.Window(width,height,'2D gravity simulator')
win.set_vsync(False)

batch=pyglet.graphics.Batch()

#Showing labels
fpstxt = pyglet.text.Label('Welcome!',font_name ='Cooper',font_size = 12,x = 10, y = win.height-10,anchor_x ='left', anchor_y ='top')
pausetxt = pyglet.text.Label('||',font_name ='Cooper',font_size = 12,x = 10, y = win.height-25,anchor_x ='left', anchor_y ='top')
infotxt = pyglet.text.Label('I : Open/Close Info\nP : Pause/Play\nW/A/S/D : Camera movement\nT-G : Calculation mode\nY-H and U-J : Change set scale\nN : Add new object',font_name ='Cooper',font_size = 12,x = width-10, y = height-10,anchor_x ='right', anchor_y ='top',multiline=True,width=500,align='right')

#Other variables
isPlaying=False
isAddingObject=False
pendingNewObject=False
isAddingMassAndRadius=False
velocityBasedColor=False
mode=0

showInfo=True
tempScaleX=1
tempScaleY=1

#camera position
camx=0
camy=0
camZoom=1


newObjX=0
newObjY=0
newObjVX=0
newObjVY=0
newObjM=0
newObjR=1

#Circles will be rendered
circles=[]



#key events
keys={'w':False,'s':False,'a':False,'d':False,'q':False,'e':False}
# temp = pyglet.shapes.Circle(0,0,500,color=(255,255,255),batch=batch)
# circles.append(temp)


# gs.addObject(width/2+100,height/2,0,-1,1e+15,5)
# gs.addObject(width/2-100,height/2,0,10,1e+14,5)

gs.addObject(width/2,height/2,0,0,1e+15,25)
# gs.addRandomObjects(200,mr=1e+12)
gs.addRandomObjects(256,vxr=1,vyr=1,mr=1e+12,rr=6)

gs.syncNumpy()
gs.syncCUDA()



    

@win.event
def on_draw():
    win.clear()
    fpstxt.draw()
    if not isPlaying:
        pausetxt.draw()
    if showInfo:
        infotxt.draw()
    batch.draw()
    
def updateOnPyData():
    i=0
    while i<len(gs.pyData.x):
        rx=(gs.pyData.x[i]-camx)*camZoom
        ry=(gs.pyData.y[i]-camy)*camZoom
        rr=gs.pyData.r[i]*camZoom
        
        if rx+rr>0 and rx-rr<width and ry+rr>0 and ry-rr<width:
            if velocityBasedColor:
                ix=abs(gs.pyData.vx[i]/(abs(gs.pyData.vx[i])+10))
                iy=abs(gs.pyData.vy[i]/(abs(gs.pyData.vy[i])+10))
                intensity=((gs.pyData.vx[i]**2)+(gs.pyData.vy[i]**2))**0.5
                
                intensity=intensity/(intensity+10)
                cb=int(255*(1-intensity))
                cg=int(127*ix*intensity)
                cr=int(127*iy*intensity)
                
                temp = pyglet.shapes.Circle(rx,ry,rr,color=(cr,cg,cb),batch=batch)
                
            else:
                temp = pyglet.shapes.Circle(rx,ry,rr,color=(255,255,255),batch=batch)
            circles.append(temp)
            
        i+=1 

def updateOnNpData():
    i=0
    while i<gs.npData.x.shape[0]:
        rx=(gs.npData.x[0,i]-camx)*camZoom
        ry=(gs.npData.y[0,i]-camy)*camZoom
        rr=gs.npData.r[i]*camZoom
        
        if rx+rr>0 and rx-rr<width and ry+rr>0 and ry-rr<width:
            if velocityBasedColor:
                ix=abs(gs.npData.vx[i]/(abs(gs.npData.vx[i])+10))
                iy=abs(gs.npData.vy[i]/(abs(gs.npData.vy[i])+10))
                intensity=((gs.npData.vx[i]**2)+(gs.npData.vy[i]**2))**0.5
                
                intensity=intensity/(intensity+10)
                cb=int(255*(1-intensity))
                cg=int(127*ix*intensity)
                cr=int(127*iy*intensity)
                
                temp = pyglet.shapes.Circle(rx,ry,rr,color=(cr,cg,cb),batch=batch)
                
            else:
                temp = pyglet.shapes.Circle(rx,ry,rr,color=(255,255,255),batch=batch)
            circles.append(temp)
            
        i+=1
        
def updateOnCUDAData():
    i=0
    while i<gs.cudaData.x.shape[0]:
        rx=(gs.cudaData.x[0,i]-camx)*camZoom
        ry=(gs.cudaData.y[0,i]-camy)*camZoom
        rr=gs.cudaData.r[i]*camZoom
        
        if rx+rr>0 and rx-rr<width and ry+rr>0 and ry-rr<width:
            if velocityBasedColor:
                ix=abs(gs.cudaData.vx[i]/(abs(gs.cudaData.vx[i])+10))
                iy=abs(gs.cudaData.vy[i]/(abs(gs.cudaData.vy[i])+10))
                intensity=((gs.cudaData.vx[i]**2)+(gs.cudaData.vy[i]**2))**0.5
                
                intensity=intensity/(intensity+10)
                cb=int(255*(1-intensity))
                cg=int(127*ix*intensity)
                cr=int(127*iy*intensity)
                
                temp = pyglet.shapes.Circle(rx,ry,rr,color=(cr,cg,cb),batch=batch)
                
            else:
                temp = pyglet.shapes.Circle(rx,ry,rr,color=(255,255,255),batch=batch)
            circles.append(temp)
            
        i+=1

updateFunc={0:updateOnPyData,1:updateOnNpData,2:updateOnNpData,3:updateOnNpData,4:updateOnNpData,5:updateOnNpData}
def update(frame_time):
    global camx,camy,camZoom,mode  
    
    if isPlaying:
        if frame_time != 0:
            
            #update fps meter
            fpstxt.text = gravitySim.Modes.modes[mode] +' '+str(round(1/frame_time,3))+' fps'
            
            #calculate next time period
            
            gs.calc(frame_time,mode=mode)

            
    
    
    
    #clearing all object every frame
    circles.clear()
    
    
    updateFunc[mode]()
   
    if pendingNewObject:
        rx=(newObjX-camx)*camZoom
        ry=(newObjY-camy)*camZoom
        rr=newObjR*camZoom
        temp = pyglet.shapes.Circle(rx,ry,rr,color=(255,255,255),batch=batch)
        circles.append(temp)
        
                
    #pan camera
    if not isAddingObject:
        
        #Camera movement
        if keys['w']:         
            camy+=100*frame_time/camZoom
            
        if keys['s']:         
            camy-=100*frame_time/camZoom
            
        if keys['a']:         
            camx-=100*frame_time/camZoom
            
        if keys['d']:         
            camx+=100*frame_time/camZoom
                 
        #Camera zoom
        if keys['q']:   
            camZoom/=1+(0.1*frame_time)     
            camx-=width*(0.1*frame_time)/camZoom/2
            camy-=height*(0.1*frame_time)/camZoom/2                         
            
        if keys['e']:         
            camZoom*=1+(0.1*frame_time)
            camx+=width*(0.1*frame_time)/camZoom/2
            camy+=height*(0.1*frame_time)/camZoom/2 
        
            
            
        
# key press event     
@win.event 
def on_key_press(symbol, modifier): 
    global isPlaying,isAddingObject,camx,camy,camZoom,tempScaleX,tempScaleY,showInfo,mode,velocityBasedColor
    
    if symbol == pyglet.window.key.N:         
        if not isPlaying:
            isAddingObject = not isPlaying
        
    if symbol == pyglet.window.key.P:         
        isPlaying = not isPlaying
        
    if symbol == pyglet.window.key.C:         
        velocityBasedColor = not velocityBasedColor
        
    if symbol == pyglet.window.key.G:         
        mode=(mode+1)%gravitySim.Modes.length
        
    if symbol == pyglet.window.key.T:         
        mode=(mode-1)%gravitySim.Modes.length
        
    if symbol == pyglet.window.key.I:         
        showInfo = not showInfo
        
    if symbol == pyglet.window.key.Y:
        tempScaleX*=10
        
    if symbol == pyglet.window.key.H:         
        tempScaleX/=10
        
    if symbol == pyglet.window.key.U:
        tempScaleY*=10
        
    if symbol == pyglet.window.key.J:         
        tempScaleY/=10
      
        
    #Key down check
    if symbol == pyglet.window.key.W:         
        keys['w']=True
        
    if symbol == pyglet.window.key.S:         
        keys['s']=True
    
    if symbol == pyglet.window.key.A:         
        keys['a']=True
    
    if symbol == pyglet.window.key.D:         
        keys['d']=True
        
    if symbol == pyglet.window.key.Q:         
        keys['q']=True
        
    if symbol == pyglet.window.key.E:         
        keys['e']=True
        
    
  
@win.event 
def on_key_release(symbol, modifiers):
    
    #Key down check
    if symbol == pyglet.window.key.W:                 
        keys['w']=False
        
    if symbol == pyglet.window.key.S:         
        keys['s']=False
    
    if symbol == pyglet.window.key.A:         
        keys['a']=False
    
    if symbol == pyglet.window.key.D:         
        keys['d']=False
        
    if symbol == pyglet.window.key.Q:         
        keys['q']=False
    
    if symbol == pyglet.window.key.E:         
        keys['e']=False
        
#mouse event
@win.event
def on_mouse_press(x, y, button, modifier):
    global newObjX,newObjY,newObjM,newObjR,pendingNewObject,isAddingMassAndRadius,isAddingObject
    if button == pyglet.window.mouse.LEFT:
        if isAddingObject:
            if isAddingMassAndRadius:
                newObjM=(width-x)*tempScaleX
                newObjR=(height-y)*tempScaleY
                isAddingMassAndRadius=False
                isAddingObject=False
                pendingNewObject=False
                gs.addObject(newObjX, newObjY, newObjVX, newObjVY, newObjM, newObjR)
            else:
                newObjX=x
                newObjY=y
                pendingNewObject=True
@win.event            
def on_mouse_release(x, y, button, modifiers):
    global pendingNewObject,newObjX,newObjY,newObjVX,newObjVY,isAddingMassAndRadius
    if button == pyglet.window.mouse.LEFT:
        if pendingNewObject and not isAddingMassAndRadius:            
            newObjVX=x-newObjX
            newObjVY=y-newObjY
            isAddingMassAndRadius=True            
        
@win.event
def on_window_close(win):
    pyglet.clock.unschedule(update)
    pyglet.app.exit()
        

pyglet.clock.schedule(update)
pyglet.app.run()
