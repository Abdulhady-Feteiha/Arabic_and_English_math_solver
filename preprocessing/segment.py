
import cv2 as cv
from config import processed_path
import glob
import os
import numpy as np

def get_contours(image) :

image = cv2.resize(image,(300,500), interpolation = cv2.INTER_AREA)
cv2_imshow(image)
# Save a copy of the original image,
orig = image.copy()
# Convert the from RGB to gray, and Blur the image
grayImageBlur = cv2.blur(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY),(3,3))
# Apply canny edge detector then find contours
edges = cv2.Canny(grayImageBlur, 100, 300, 3)
# Get the contours of the paper
Contours = cv2.findContours(edges.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
Contours = imutils.grab_contours(Contours)
Contours = sorted(Contours, key=cv2.contourArea, reverse=True)[:1]
perimeter = cv2.arcLength(Contours[0], True)
paper_borders = cv2.approxPolyDP(Contours[0], 0.02*perimeter, True)
# cv2.drawContours(image, [paper_borders], -1, (0,255,0), 2)
# Need to make sure the the formed contours are able to form a rectangle
paper_sides = np.zeros((4,2),np.float32)
if paper_borders.shape[0] == 4 :
  paper_borders = paper_borders.reshape(4,2)
  s = np.sum(paper_borders, axis=1)
  paper_sides[0] = paper_borders[np.argmin(s)]
  paper_sides[2] = paper_borders[np.argmax(s)]
  diff = np.diff(paper_borders, axis=1)
  paper_sides[1] = paper_borders[np.argmin(diff)]
  paper_sides[3] = paper_borders[np.argmax(diff)]
  (s_1, s_2, s_3, s_4) = paper_sides
  width_1 = np.sqrt((s_1[0] - s_2[0])**2 + (s_1[1] - s_2[1])**2 )
  width_2 = np.sqrt((s_4[0] - s_3[0])**2 + (s_4[1] - s_3[1])**2 )
  maxWidth = max(int(width_1), int(width_2))
  height_1 = np.sqrt((s_1[0] - s_4[0])**2 + (s_1[1] - s_4[1])**2 )
  height_2 = np.sqrt((s_2[0] - s_3[0])**2 + (s_2[1] - s_3[1])**2 )
  maxHeight = max(int(height_1), int(height_2))
  dst = np.array([[0,0],[maxWidth-1, 0],[maxWidth-1, maxHeight-1],[0, maxHeight-1]],np.float32)
  transformMatrix = cv2.getPerspectiveTransform(paper_sides, dst)
  scan = cv2.warpPerspective(orig, transformMatrix, (maxWidth, maxHeight))
  cv2_imshow(scan)
  return scan
else :
  # More or less than 4 contours were detected, which means the results are inaccurate
  # In this case, the user needs to retake the image
  print('Unable to detect paper borders, please try to retake the image')
  return 'Unable to detect paper borders'



def segment(img_path,digital, scan):
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
        if scan :
          img = get_contours(img)
          if type(img) == str :
            message = 'Sorry, could not identify image borders, please try again.'
            return message
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
