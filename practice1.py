import numpy as np
import cv2
import sys


def rgb():
    imgfile = '/Users/a/Desktop/dot.png'
    img = cv2.imread(imgfile,1)
    
def showImage():
    imgfile = '/Users/a/face_project/Practice/ex1.png'
    imgfile2 = '/Users/a/face_project/Practice/ex2.png'
    #img = cv2.imread(imgfile, cv2.IMREAD_COLOR)
    img = cv2.imread(imgfile, 1)
    img2 = cv2.imread(imgfile2, 1)

    sub_img = cv2.resize(img, (750,750))
    sub_img2 = cv2.resize(img2, (750,750))

    add1 = sub_img + sub_img2
    add2 = cv2.add(sub_img,sub_img2)

    cv2.imshow('add1',add1)
    cv2.imshow('add2',add2)

    cv2.imshow('ex2',img2)
    px = img[340,200]
    print(px)
    #img2 = cv2.imread(imgfile, cv2.IMREAD_GRAYSCALE)
    B = img.item(340,200,0)
    G = img.item(340,200,1)
    R = img.item(340,200,2)

    BGR = [B,G,R]
    print(BGR)

    print(img.shape)
    print(img.size)
    print(img.dtype)

    cv2.imshow('original',img)

    subimg = img[300:400, 350:750]
    cv2.imshow('cutting',subimg)

    img[300:400, 0:400] = subimg

    #b, g, r = cv2.split(img)
    b = img[:,:,0]
    g = img[:,:,1]
    img[:,:,2] =0
    r = img[:,:,2]

    cv2.imshow('b',b)

    print(img[100, 100])
    print(b[100,100], g[100,100], r[100,100])

    merged_img = cv2.merge((b,g,r))
    cv2.imshow('merge',merged_img)

    cv2.imshow('mix', img)
    #cv2.imshow('model2',img2)
    cv2.namedWindow('ex1',cv2.WINDOW_NORMAL)
    cv2.imshow('ex1',img)
    cv2.waitKey(0) 
    cv2.destroyAllWindows() 

#showImage()

#def onMouse(x):
#    pass

def imgBlending(img1_f,img2_f):
    img1_o = cv2.imread(img1_f)
    img2_o = cv2.imread(img2_f)

    img1 = cv2.resize(img1_o,(960,540))
    img2 = cv2.resize(img2_o,(960,540))

    cv2.namedWindow('ImgPane')
    cv2.createTrackbar('MIXING','ImgPane',0,100,onMouse)
    mix = cv2.getTrackbarPos('MIXING','ImgPane')

    while True:
        img = cv2.addWeighted(img1, float(100-mix)/100, img2, float(mix)/100,0)
        cv2.imshow('ImgPane', img)

        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break
        mix = cv2.getTrackbarPos('MIXING','ImgPane')

    cv2.destroyAllWindows()
##imgBlending('/Users/a/face_project/Practice/ex1.png','/Users/a/face_project/Practice/ex2.png')

def bitOperation(hpos, vpos):
    img1 = cv2.imread('/Users/a/face_project/Practice/ex1.png')
    img2 = cv2.imread('/Users/a/face_project/Practice/ex2.png')

    rows, cols, channels = img2.shape
    roi = img1[vpos:rows+vpos, hpos:cols+hpos]

    img2gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)

    img1_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
    img2_fg = cv2.bitwise_and(img2, img2, mask=mask)

    dst = cv2.add(img1_bg, img2_fg)
    img1[vpos:rows+vpos, hpos:cols+hpos] = dst

    cv2.imshow('result', img1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#bitOperation(10,10)

def hsv():
    blue = np.uint8([[[255,0,0]]])
    green = np.uint8([[[0,255,0]]])
    red = np.uint8([[[0,0,255]]])

    hsv_blue = cv2.cvtColor(blue, cv2.COLOR_BGR2HSV)
    hsv_green = cv2.cvtColor(green, cv2.COLOR_BGR2HSV)
    hsv_red = cv2.cvtColor(red, cv2.COLOR_BGR2HSV)

    print('HSV for BLUE: ', hsv_blue)
    print('HSV for GREEN: ', hsv_green)
    print('HSV for RED: ', hsv_red)

#hsv()

def tracking():
    try:
        print('camera on.')
        cap = cv2.VideoCapture(0)
    except:
        print('false')
        return
    while True:
        ret, frame = cap.read()

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        lower_blue = np.array([110,100,100])
        upper_blue = np.array([130,255,255])
        
        lower_green = np.array([50,100,100])
        upper_green = np.array([70,255,255])
        
        lower_red = np.array([-10,100,100])
        upper_red = np.array([10,255,255])

        mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
        mask_green = cv2.inRange(hsv, lower_green, upper_green)
        mask_red = cv2.inRange(hsv, lower_red, upper_red)

        res1 = cv2.bitwise_and(frame, frame, mask=mask_blue)
        res2 = cv2.bitwise_and(frame, frame, mask=mask_green)
        res3 = cv2.bitwise_and(frame, frame, mask=mask_red)

        cv2.imshow('original',frame)
        cv2.imshow('BLUE', res1)
        cv2.imshow('GREEN', res2)
        cv2.imshow('RED', res3)

        k=cv2.waitKey(1) & 0xFF
        if k == 27:
            break

    cv2.destroyAllWindows()

#tracking()

def threshold():
    img = cv2.imread('/Users/a/Desktop/ex1.jpeg',cv2.IMREAD_GRAYSCALE)

    ret, thr1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    ret, thr2 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
    ret, thr3 = cv2.threshold(img, 127, 255, cv2.THRESH_TRUNC)
    ret, thr4 = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO)
    ret, thr5 = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO_INV)

    cv2.imshow('original', img)
    cv2.imshow('BINARY', thr1)
    cv2.imshow('BINARY_INV',thr2)
    cv2.imshow('TRUNC',thr3)
    cv2.imshow('TOZERO', thr4)
    cv2.imshow('TOZERO_INV', thr5)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
#threshold()

def adt_threshold():
    img = cv2.imread('/Users/a/Desktop/ex1.jpeg',cv2.IMREAD_GRAYSCALE)

    ret, thr1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    thr2 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    thr3 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    titles = ['original', 'Global Thresholding(v=127)','Adaptive MEAD','Adaptive GAUSSIAN']
    images = [img, thr1, thr2, thr3]

    for i in range(4):
        cv2.imshow(titles[i], images[i])

    cv2.waitKey(0)
    cv2.destroyAllWindows()

#adt_threshold()

def transform():
    img = cv2.imread('/Users/a/Desktop/ex1.jpeg')
    h, w= img.shape[:2]
    
    img2 = cv2.resize(img, None, fx=0.5, fy=1, interpolation = cv2.INTER_AREA)
    img3 = cv2.resize(img, None, fx=1, fy=0.5, interpolation = cv2.INTER_AREA)
    img4 = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation = cv2.INTER_AREA)

    cv2.imshow('original',img)
    cv2.imshow('fx=0.5',img2)
    cv2.imshow('fy=0.5',img3)
    cv2.imshow('fx=0.5,fy=0.5',img4)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

#transform()

def transform():
    img = cv2.imread('/Users/a/Desktop/ex1.jpeg')
    h, w=img.shape[:2]
    print(h)
    print(w)
    M = np.float32([[1,0,100],[0,1,50]])

    img2 = cv2.warpAffine(img, M, (w,h))
    cv2.imshow('shift image',img2)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

#transform()

def Rtransform():
    img = cv2.imread('/Users/a/Desktop/ex1.jpeg')
    h, w = img.shape[:2]

    M1 = cv2.getRotationMatrix2D((w/2,h/2),45,1)
    M2 = cv2.getRotationMatrix2D((w/2,h/2), 90,1)

    img2 = cv2.warpAffine(img,M1,(w,h))
    img3 = cv2.warpAffine(img, M2,(w,h))

    cv2.imshow('45',img2)
    cv2.imshow('90',img3)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
#Rtransform()

def Atransform():
    img = cv2.imread('/Users/a/Desktop/ex1.jpeg')
    h, w = img.shape[:2]

    pts1 = np.float32([[50,50],[200,50],[20,200]])
    pts2 = np.float32([[10,100],[200,50],[100,250]])

    M = cv2.getAffineTransform(pts1, pts2)

    img2 = cv2.warpAffine(img,M,(w,h))

    cv2.imshow('original',img)
    cv2.imshow('affine-transform',img2)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

#Atransform()



#Ptransform()

def bluring():
    img = cv2.imread('/Users/a/Desktop/ex1.jpeg')
    
    kernel = np.ones((5,5),np.float32)/25
    blur = cv2.filter2D(img,-1,kernel)

    cv2.imshow('org',img)
    cv2.imshow('blur',blur)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

#bluring()

def callBackFunc(event, x, y, flags, userdata):
    print('www')
    rgb_val=0
    if event == cv2.EVENT_LBUTTONDOWN:
        print('aaa')
        if rgb_val==1:
            print('rgb_val=0')
            rgb_val = 0
        else:
            print('rgb_val=1')
            rgb_val = 1
        img = cv2.imread(var1,rgb_val)

def Ptransform():
    img = cv2.imread('/Users/a/Desktop/ex1.jpeg')
    h, w = img.shape[:2]

    pts1 = np.float32([[0,0],[300,0],[0,300],[300,300]])
    pts2 = np.float32([[56,65],[368,52],[28,387],[389,390]])

    M = cv2.getPerspectiveTransform(pts1, pts2)

    img2 = cv2.warpPerspective(img,M,(w,h))

    cv2.imshow('original',img)
    cv2.imshow('Perspective-transform',img2)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


        
def image_viewer():
 #   mat_gray = []   
    global ix,iy
    print(ix,iy)
    var1 = sys.argv[1]
    img = cv2.imread(var1)
    #cv2.cvtColor(img,mat_gray,CV_BGR2GRAY)
    cv2.namedWindow('ex1')
    #cv2.imshow('ex1',img)
    #cv2.imshow('ex2',mat_gray)
    cv2.setMouseCallback('ex1',callBackF,param=img)
    while True:
        #img = cv2.imread(var1,rgb_val)
        cv2.imshow('ex1',img)
        k = cv2.waitKey(0)
        if k == 27:
            break

    
  #  cv2.destroyAllWindows()
mouse_x1,mouse_y1,mouse_x2,mouse_y2=0,0,50,50
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
    return sub_img

def imageload():
    global mouse_x1,mouse_y1,mouse_x2,mouse_y2
    var1 = sys.argv[1]
    #var1 = cv2.resize(var1,(500,500))
    original = cv2.imread(var1,1)
    gray = cv2.imread(var1,0)
    NORMAL = cv2.WINDOW_NORMAL
    AUTO = cv2.WINDOW_AUTOSIZE
    b=original
    c= NORMAL
    cv2.namedWindow('imgloadview',c)
    cv2.setMouseCallback('imgloadview',callBackF,param=original)
    

    #cv2.namedWindow('practice2',cv2.WINDOW_AUTOSIZE)
    while True:
        
        cv2.imshow('imgloadview',b)
        original_x,original_y,original_a=b.shape
        print(b.shape)
        k = cv2.waitKey(0)
        if k == 27:
            cv2.destroyAllWindows()
            break
        elif k == ord('g'):
            b=gray
        elif k == ord('r'):
            b=original
        elif k == ord('a'):
            c=AUTO
            cv2.destroyAllWindows()
        elif k == ord('n'):
            c=NORMAL
            cv2.destroyAllWindows()
        elif k == ord('c'):
            sub=Sub_img(original)
            b=cv2.resize(sub,(original_x,original_y))
        elif k == ord('.'):
            mouse_x1,mouse_x2 = mouse_x1+10,mouse_x2+10
            sub=Sub_img(original)
            b=cv2.resize(sub,(original_x,original_y))
        elif k == ord(','):
            mouse_x1,mouse_x2 = mouse_x1-10,mouse_x2-10
            sub=Sub_img(original)
            b=cv2.resize(sub,(original_x,original_y))
        elif k == ord(';'):
            mouse_y1,mouse_y2 = mouse_y1+10,mouse_y2+10
            sub=Sub_img(original)
            b=cv2.resize(sub,(original_x,original_y))
        elif k == ord('l'):
            mouse_y1,mouse_y2 = mouse_y1-10,mouse_y2-10
            sub=Sub_img(original)
            b=cv2.resize(sub,(original_x,original_y)) 
            
imageload()