
import cv2
import cv
import numpy as np
import matplotlib.image as mpimg
from matplotlib import pyplot as plt
 
def threshold_slow(T, image):
    # grab the image dimensions
    h = image.shape[0]
    w = image.shape[1]
    d = image.shape[2]
    # loop over the image, pixel by pixel
    for y in range(0, h):
        for x in range(0, w):
            for z in range(0, d):
                # threshold the pixel
                
                if image[y, x,z] >= T:
                    image[y, x,z] = 255
                else:
                    image[y, x,z] = 0
            
    # return the thresholded image
    return image
def grab_frame(cam):
    




    #cv2.namedWindow("test")

    #img_counter = 0

    while True:
        ret, color1 = cam.read()
        #r = 100.0 / color1.shape[1]
        r = 640.0 / color1.shape[1]
        #r = 0.25
        dim = (100, int(color1.shape[0] * r))
        dim = (640,480)
        # perform the actual resizing of the image and show it
        color = cv2.resize(color1, dim, interpolation = cv2.INTER_AREA)
        
       
        #color = color1.copy()
        b = color.copy()
        # set green and red channels to 0
        b[:, :, 1] = 0
        b[:, :, 2] = 0


        g = color.copy()
        # set blue and red channels to 0
        g[:, :, 0] = 0
        g[:, :, 2] = 0

        r = color.copy()
        # set blue and green channels to 0
        r[:, :, 0] = 0
        r[:, :, 1] = 0
        
        #y= color.copy()
        #gray = cv2.cvtColor(l,cv2.COLOR_RGB2GRAY)
        #_,y = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY)
        #y = cv2.cvtColor(y, cv2.COLOR_GRAY2RGB)
        
        y = cv2.add(r,g)
        
        d = color.copy()
        gray1 = cv2.cvtColor(d,cv2.COLOR_RGB2GRAY)
        _,p = cv2.threshold(gray1, 60, 255, cv2.THRESH_BINARY)
        p = cv2.cvtColor(p, cv2.COLOR_GRAY2RGB)
        #threshold_slow(220,p)
        return [color,b,g,r,y,p]


cam = cv2.VideoCapture(0)
    
    #cv2.waitKey(0)
while(1):
    ret, color = cam.read()
    [color,b,g,r,y,p] = grab_frame(cam)
    horiz = np.hstack((color,b,g))
    #verti = np.vstack((color,r))
    horiz1 = np.hstack((r,y,p))
    verti = np.vstack((horiz,horiz1))
    cv2.imshow('HORIZONTAL', verti)
    
    if not ret:
        break
    k = cv2.waitKey(1)

    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    
        
cam.release()

cv2.destroyAllWindows()
   
       
    

