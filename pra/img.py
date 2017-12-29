import numpy as np
import cv2
import sys
import os

mouse_x1,mouse_y1,mouse_x2,mouse_y2,nextfile=0.,0.,0.,0.,1
basename=[]
drawing = False
capture = False
drag = False
drag_x,drag_y,original__x,original__y=0.,0.,0.,0.
crr_x1,crr_x2,crr_y1,crr_y2=0.,0.,0.,0.
pro=2
def init():
    global mouse_x1,mouse_y1,mouse_x2,mouse_y2,drawing,capture,drag,drag_x,drag_y,original__x,original__y,crr_x1,crr_x2,crr_y1,crr_y2
    mouse_x1,mouse_y1,mouse_x2,mouse_y2=0.,0.,0.,0.
    drawing = False
    capture = False
    drag = False
    drag_x,drag_y,original__x,original__y=0.,0.,0.,0.
    crr_x1,crr_x2,crr_y1,crr_y2=0.,0.,0.,0.
def allfiles(path):
    global nextfile, basename
    res = []

    for root, dirs, files in os.walk(path):
        rootpath = os.path.join(os.path.abspath(path))
        #print(rootpath)
        
        for file in files:
            filepath = os.path.join(rootpath, file)
            #print(rootpath)
            base = os.path.basename(filepath)
            #print(filepath)
            res.append(filepath)
        i=0
        for file in files:
            filepath = os.path.join(rootpath, file)
            base = os.path.basename(filepath)
            #print(base)
            #print(basename)
            if basename == base :
                break
            else:
                i = i+1
        nextfile = i
        #print(nextfile)

    return res
def callBackF(event, x, y, flags, userdata):
    #print(x,y)
    global mouse_x1,mouse_y1,mouse_x2,mouse_y2,drawing,capture,drag_x,drag_y,original__x,original__y,drag,crr_x1,crr_x2,crr_y1,crr_y2,pro
    #data_ = cv2.imread(userdata,1)
    #data = cv2.resize(data_,(400,400))
    
    data_ = cv2.imread(userdata,1)
    data = cv2.resize(data_,(pro*original__y,pro*original__x))
    ex=False   
    x_,y_=0.,0.
    if event==cv2.EVENT_LBUTTONDOWN:
        #print(capture)
        if capture == False:
            drawing = True
            mouse_x1 = x
            mouse_y1 = y
            #print(mouse_x1,mouse_y1)
        elif capture == True:
            drag_x,drag_y=x,y
            drag = True
            #print(drag_x,drag_y)
            #print(drag)
        
    elif event==cv2.EVENT_LBUTTONUP:
        drawing = False
        drag = False
        if capture == False:
            mouse_x2 = x
            mouse_y2 = y
            cv2.rectangle(data,(int(mouse_x1),int(mouse_y1)),(x,y),(255,0,0),1)
        else :
            mouse_x1,mouse_x2,mouse_y1,mouse_y2=crr_x1,crr_x2,crr_y1,crr_y2
        #print(mouse_x2,mouse_y2)

    elif event == cv2.EVENT_MOUSEMOVE:
        
        if drawing == True:
            cv2.rectangle(data,(int(mouse_x1),int(mouse_y1)),(x,y),(255,0,0),1)
            #if drag == True:
        elif drawing == False:
            ex=True
        #print("move")
        #print(capture,drag)
        if capture==True and drag == True:
            #print(capture,drag)
            x_,y_ = (x-drag_x),(y-drag_y)
            #print(x_,y_)
            mouse_x1,mouse_x2 = mouse_x1+x_,mouse_x2+x_
            mouse_y1,mouse_y2 = mouse_y1+y_,mouse_y2+y_
            #print(mouse_x1,mouse_x2,mouse_y1,mouse_y2,original__x,original__y)
            if mouse_x1>0 and mouse_x2<pro*original__y and mouse_y1>0 and mouse_y2<pro*original__x:
                sub=Sub_img(data)
                data=cv2.resize(sub,(pro*original__y,pro*original__x))
                cv2.imshow('imgloadview',data)
                crr_x1,crr_x2,crr_y1,crr_y2=mouse_x1,mouse_x2,mouse_y1,mouse_y2
            mouse_x1,mouse_x2 = mouse_x1-x_,mouse_x2-x_
            mouse_y1,mouse_y2 = mouse_y1-y_,mouse_y2-y_
            #print(mouse_x1,mouse_x2,mouse_y1,mouse_y2,original__x,original__y)
        if ex==True:
            return
    if capture == False:
        cv2.imshow('imgloadview',data)            

def Sub_img(original_):
    global mouse_x1,mouse_y1,mouse_x2,mouse_y2
    sub_img = original_[int(mouse_y1):int(mouse_y2),int(mouse_x1):int(mouse_x2)]
    #sub_img = original_[mouse_x1:mouse_x2,mouse_y1:mouse_y2]
    return sub_img
def onChange(x):
    pass

    
def imageload():
    global mouse_x1,mouse_y1,mouse_x2,mouse_y2,nextfile,basename,original__x,original__y,capture,pro
    out=0
    res = []
    basename = os.path.basename(sys.argv[1])
    #print(basename)
    #print(os.path.dirname(sys.argv[1]))
    dir = os.path.dirname(sys.argv[1])
    res = allfiles(dir)
    dirsize = len(res)
    
    #print(dirsize)
    #print(res)
    while True:
   
        var1 = res[nextfile]
        original = cv2.imread(var1,1)
        original__x,original__y=original.shape[:2]
        original_=cv2.resize(original,(pro*original__y,pro*original__x))
        gray = cv2.cvtColor(original_,cv2.COLOR_BGR2GRAY)
        NORMAL = cv2.WINDOW_NORMAL
        AUTO = cv2.WINDOW_AUTOSIZE
        b=original_
        window= NORMAL
        #print("first while")
        
        while True:
            
            #print("second while")
            cv2.namedWindow('imgloadview',window)
            #cv2.resizeWindow('imgloadview',700,700)
            cv2.setMouseCallback('imgloadview',callBackF,param=var1)
            cv2.imshow('imgloadview',b)
            #original__x,original__y=b.shape[:2]
            print(b.shape)
            k = cv2.waitKey(0)
            if k == 27:
                cv2.destroyAllWindows()
                out=1
                break
            elif k == ord('g'):
                b=gray
            elif k == ord('h'):
                b=original_
                init()
            elif k == ord('a'):
                window=AUTO
                cv2.destroyAllWindows()
            elif k == ord('s'):
                window=NORMAL
                cv2.destroyAllWindows()
            elif k == ord('c'):
                sub=Sub_img(original_)
                b=cv2.resize(sub,(pro*original__y,pro*original__x))
                capture = True
                print('bbox size : ',mouse_x2-mouse_x1,mouse_y2-mouse_y1)
            
            elif k == ord(']'):
                if nextfile < dirsize-1 :
                    nextfile = nextfile+1
                    init()
                    break
                elif nextfile == dirsize-1 :
                    nextfile = 1
                    init()
                    break
            elif k == ord('['):
                if nextfile > 1 :
                    nextfile = nextfile-1
                    init()
                    break
                elif nextfile == 1 :
                    nextfile = dirsize-1
                    init()
                    break
        cv2.destroyAllWindows()
        if out==1:
            break

            
imageload()