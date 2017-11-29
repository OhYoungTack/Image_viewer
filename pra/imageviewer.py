import numpy as np
import cv2
import sys
import os

mouse_x1,mouse_y1,mouse_x2,mouse_y2,nextfile=0,0,50,50,1

def allfiles(path):
    res = []

    for root, dirs, files in os.walk(path):
        rootpath = os.path.join(os.path.abspath(path), root)

        for file in files:
            filepath = os.path.join(rootpath, file)
            res.append(filepath)

    return res
def callBackF(event, x, y, flags, userdata):
    #print(x,y)
    global mouse_x1,mouse_y1,mouse_x2,mouse_y2
    if event==cv2.EVENT_LBUTTONDOWN:
        mouse_x1 = x
        mouse_y1 = y
        print(mouse_x1,mouse_y1)
    elif event==cv2.EVENT_LBUTTONUP:
        mouse_x2 = x
        mouse_y2 = y
        print(mouse_x2,mouse_y2)
        
    

def Sub_img(original):
    global mouse_x1,mouse_y1,mouse_x2,mouse_y2
    sub_img = original[mouse_y1:mouse_y2,mouse_x1:mouse_x2]
    #sub_img = original[mouse_x1:mouse_x2,mouse_y1:mouse_y2]
    return sub_img
def onChange(x):
    pass
def imageload():
    global mouse_x1,mouse_y1,mouse_x2,mouse_y2,nextfile
    out=0
    res = []
    res = allfiles(sys.argv[1])
    dirsize = len(res)
    print(dirsize)
    print(res)
    while True:
        var1 = res[nextfile]#res[nextfile]
    #var1 = cv2.resize(var1,(500,500))
        original = cv2.imread(var1,1)
        gray = cv2.cvtColor(original,cv2.COLOR_BGR2GRAY)
        NORMAL = cv2.WINDOW_NORMAL
        AUTO = cv2.WINDOW_AUTOSIZE
        b=original
        c= NORMAL
    #cv2.namedWindow('imgloadview',c)
    #cv2.setMouseCallback('imgloadview',callBackF,param=original)
    

    #cv2.namedWindow('practice2',cv2.WINDOW_AUTOSIZE)
        while True:
            cv2.namedWindow('imgloadview',c)
            cv2.setMouseCallback('imgloadview',callBackF,param=original)
            cv2.imshow('imgloadview',b)
            original_x,original_y=b.shape[:2]
            #cv2.createTrackbar('X','imgloadview',0,original_x,onChange)
            #cv2.createTrackbar('Y','imgloadview',0,original_y,onChange)
            print(b.shape)
            k = cv2.waitKey(0)
            if k == 27:
                cv2.destroyAllWindows()
                out=1
                break
            elif k == ord('g'):
                b=gray
            elif k == ord('h'):
                b=original
            elif k == ord('a'):
                c=AUTO
                cv2.destroyAllWindows()
            elif k == ord('s'):
                c=NORMAL
                cv2.destroyAllWindows()
            elif k == ord('c'):
                sub=Sub_img(original)
                b=cv2.resize(sub,(original_y,original_x))
                print('bbox size : ',mouse_x2-mouse_x1,mouse_y2-mouse_y1)
            elif k == ord(','):
                #mouse_x1=cv2.getTrackbarPos('X','imageloadview')
                #mouse_x2=cv2.getTrackbarPos('X','imageloadview')
                if mouse_x1+3<original_y and mouse_x2+3<original_y :
                    mouse_x1,mouse_x2 = mouse_x1+3,mouse_x2+3
                    sub=Sub_img(original)
                    b=cv2.resize(sub,(original_y,original_x))
            elif k == ord('n'):
                if mouse_x1-3>0 and mouse_x2-1>0 :
                    mouse_x1,mouse_x2 = mouse_x1-3,mouse_x2-3
                    sub=Sub_img(original)
                    b=cv2.resize(sub,(original_y,original_x))
            elif k == ord('m'):
                if mouse_y1+3<original_x and mouse_y2+3<original_x :
                    mouse_y1,mouse_y2 = mouse_y1+3,mouse_y2+3
                    sub=Sub_img(original)
                    b=cv2.resize(sub,(original_y,original_x))
            elif k == ord('j'):
                if mouse_y1-3>0 and mouse_y2-3>0 :
                    mouse_y1,mouse_y2 = mouse_y1-3,mouse_y2-3
                    sub=Sub_img(original)
                    b=cv2.resize(sub,(original_y,original_x))
            elif k == ord(']'):
                if nextfile < dirsize-1 :
                    nextfile = nextfile+1
                    break
                elif nextfile == dirsize-1 :
                    nextfile = 1
                    break
            elif k == ord('['):
                if nextfile > 1 :
                    nextfile = nextfile-1
                    break
                elif nextfile == 1 :
                    nextfile = dirsize-1
                    break
        cv2.destroyAllWindows()
        if out==1:
            break

            
imageload()