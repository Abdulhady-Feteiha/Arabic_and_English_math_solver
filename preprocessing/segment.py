
import cv2 as cv
from config import processed_path
import glob
import os
import numpy as np

def segment(img_path,digital):
    old_imgs = glob.glob(processed_path+'/*.png')
    if old_imgs:
        for old in  old_imgs:
            os.remove(old)
    if digital:
        img = cv.imread(img_path) #load a test image
        gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY) #make sure it's gray scale
        ret,thresh = cv.threshold(gray,127,255,cv.THRESH_BINARY_INV) #invert colours and binarize
        objects, _ = cv.findContours(thresh.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE) # Find countours of objects
        objects = sorted(objects, key=lambda obj: cv.boundingRect(obj)[0]) #Extract objects bounded in boxes


        for i, obj in enumerate(objects):
            x, y, w, h = cv.boundingRect(obj) #get coordinates and shapes of object
            Digit = img[y:y+h, x:x+w] #extract an image of the object given its location
            pad = 30 # adding a pad factor to make the digit clear for the model
            if w>h: # if it's a landscape image, make sure it's tranformed into a square
              equal_pad = pad+(w-h)//2
              Digit = cv.copyMakeBorder(Digit, equal_pad, equal_pad, pad, pad, cv.BORDER_CONSTANT,value=[255,255,255])
            else:  # else if it's a portrait image, make sure it's tranformed into a square
              equal_pad = pad+(h-w)//2
              Digit = cv.copyMakeBorder(Digit, pad, pad, equal_pad, equal_pad,  cv.BORDER_CONSTANT,value=[255,255,255])

            cv.imwrite(processed_path+'/img'+str(i)+".png",Digit) #save images
    if not digital:
        img = cv.imread(img_path)
        # img = cv.resize(img,(500,300), interpolation = cv.INTER_AREA)
        # Need to consider resizing all images to a fixed size
        img = cv.resize(img, (0,0), fx=0.5, fy=0.5)
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        threshold = np.quantile(img, 0.025)
        kernel = np.ones((5,5),np.uint8)
        kernel2 = np.ones((3,3),np.uint8)
        for i in range (img.shape[0]) :
            for j in range (img.shape[1]) :
              if img[i,j] < threshold :
                img[i,j] = 255
              else :
                img[i,j] = 0
        # cv_imshow(img)
        # img = cv.morphologyEx(img, cv.MORPH_OPEN, kernel)
        img = cv.dilate(img,kernel2,iterations = 2)
        img = cv.morphologyEx(img, cv.MORPH_OPEN, kernel)
        img = abs(255-img)
        # cv_imshow(img)
        gray = img
        ret,thresh = cv.threshold(gray,127,255,cv.THRESH_BINARY_INV) #invert colours and binarize
        objects, _ = cv.findContours(thresh.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE) # Find countours of objects
        objects = sorted(objects, key=lambda obj: cv.boundingRect(obj)[0]) #Extract objects bounded in boxes
        object_areas = np.zeros((1, len(objects)))
        for i in range (0, len(objects)) :
            object_areas[0,i] = cv.contourArea(objects[i])
        object_areas = object_areas.astype(int)
        selected_objects = []
        selected_objects_indices = []
        for i, obj in enumerate(objects):
          x, y, w, h = cv.boundingRect(obj) #get coordinates and shapes of object
          area_of_obj = cv.contourArea(obj)
          if area_of_obj > int((0.1 * np.max(object_areas))) :
            selected_objects.append(obj)
            Digit = img[y:y+h, x:x+w] #extract an image of the object given its location
            pad = 30 # adding a pad factor to make the digit clear for the model
            if w>h: # if it's a landscape image, make sure it's tranformed into a square
              equal_pad = pad+(w-h)//2
              Digit = cv.copyMakeBorder(Digit, equal_pad, equal_pad, pad, pad, cv.BORDER_CONSTANT,value=[255,255,255])
            else:  # else if it's a portrait image, make sure it's tranformed into a square
              equal_pad = pad+(h-w)//2
              Digit = cv.copyMakeBorder(Digit, pad, pad, equal_pad, equal_pad,  cv.BORDER_CONSTANT,value=[255,255,255])

            cv.imwrite(processed_path+'/img'+str(i)+".png",Digit) #save images
