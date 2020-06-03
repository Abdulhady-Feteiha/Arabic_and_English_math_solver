import cv2 as cv
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import cv2 as cv

def process_dataset(X_train, X_test):
    X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
    X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')

    X_train/=255
    X_test/=255
    return X_train, X_test

def process_image(img,digital):
    img = cv.imread(img,cv.IMREAD_GRAYSCALE)
    _,inv_img = cv.threshold(img,127,255,cv.THRESH_BINARY_INV)
    if digital:
        kernel2 = np.ones((3,3),np.uint8)
        inv_img = cv.dilate(inv_img,kernel2,iterations = 1)
    res_img = cv.resize(inv_img,(28,28),interpolation = cv.INTER_AREA)
    test_img = np.reshape(res_img,(1,28,28,1))
    test_img = test_img.astype('float32')
    test_img/=255
    return test_img
