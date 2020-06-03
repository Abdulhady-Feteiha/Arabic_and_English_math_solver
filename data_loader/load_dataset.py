from config import Data_path, labels, operators, num_classes
from numpy import genfromtxt
from keras.datasets import mnist
from keras.utils import np_utils, to_categorical
from sklearn.model_selection import train_test_split
import glob
from PIL import Image, ImageOps
import numpy as np
import cv2 as cv
import os
import random

def Arabic(Data_path=Data_path, num_classes=num_classes):
    X_train = flip_and_rotate(genfromtxt(Data_path+'csvTrainImages 60k x 784.csv', delimiter=','))
    X_test = flip_and_rotate(genfromtxt(Data_path+'csvTestImages 10k x 784.csv', delimiter=','))
    y_train =  genfromtxt(Data_path+'csvTrainLabel 60k x 1.csv', delimiter=',')
    #Y_train = np_utils.to_categorical(y_train)
    y_test = genfromtxt(Data_path+'csvTestLabel 10k x 1.csv', delimiter=',')
    #Y_test = np_utils.to_categorical(y_test)
    X_train, X_test, Y_train, Y_test = data_set_mixer(X_train, X_test, y_train, y_test,num_classes)
    return X_train, X_test, Y_train, Y_test

def English(Data_path=Data_path, num_classes=num_classes):
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    number_of_classes = 10
    #Y_train = np_utils.to_categorical(y_train, number_of_classes)
    #Y_test = np_utils.to_categorical(y_test, number_of_classes)
    X_train, X_test, Y_train, Y_test = data_set_mixer(X_train, X_test, y_train, y_test,num_classes)
    return X_train, X_test, Y_train, Y_test


def Operators(Data_path, label) :
  train_images = np.empty((0,28,28), np.uint8)
  train_labels = np.empty((0,), int)
  test_images = np.empty((0,28,28), np.uint8)
  test_labels = np.empty((0,), int)
  for i in range (0, len(operators)) :
    images = read_all_images(Data_path+'operators/'+ operators[i])

    if (operators[i]=='+' or operators[i]=='-'):
      images = random.choices(images,k=7060)
    images = np.asarray(images)
    labels = np.zeros((len(images)))+label[i]
    labels = labels.astype(int)
    data_train, data_test, labels_train, labels_test = train_test_split(images, labels, test_size=0.15, shuffle = True)
    print(np.shape(data_train),np.shape(data_test))
    X_train = np.concatenate((train_images, data_train), axis = 0)
    X_test = np.concatenate((test_images, data_test), axis = 0)
    Y_train = np.append(train_labels, labels_train)
    Y_test = np.append(test_labels, labels_test)
  return X_train, X_test, Y_train, Y_test
"""
def Operators(Data_path, label) :
  train_images = np.empty((0,28,28), np.uint8)
  train_labels = np.empty((0,), int)
  test_images = np.empty((0,28,28), np.uint8)
  test_labels = np.empty((0,), int)
  for i in range (0, len(operators)) :
    if operators[i]=='1':
      images=read_all_images(Data_path+'1')
    else:
      images = read_all_images(Data_path+ operators[i])
    if (operators[i]=='+' or operators[i]=='-' or operators[i]=='1'):
      images = random.choices(images,k=10000)
    images = np.asarray(images)
    labels = np.zeros((len(images)))+label[i]
    labels = labels.astype(int)
    data_train, data_test, labels_train, labels_test = train_test_split(images, labels, test_size=0.15, shuffle = True)
    print(np.shape(data_train),np.shape(data_test))
    train_images = np.concatenate((train_images, data_train), axis = 0)
    test_images = np.concatenate((test_images, data_test), axis = 0)
    train_labels = np.append(train_labels, labels_train)
    test_labels = np.append(test_labels, labels_test)
  return train_images, test_images, train_labels, test_labels
"""
def flip_and_rotate(dataset):
  out = []
  for img in dataset:
    im = Image.fromarray(np.uint8(img.reshape(28, 28)))
    rotated = im.transpose(cv.ROTATE_90_COUNTERCLOCKWISE)
    flipped = np.asarray(ImageOps.flip(rotated))
    out.append(flipped)
  return np.asarray(out)

def read_all_images(image_dir):
  data_path = os.path.join(image_dir,'*g')
  files = glob.glob(data_path)
  all_images = []
  for f1 in files:
    img = cv.imread(f1)
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img = cv.resize(img,(28,28), interpolation = cv.INTER_AREA)
    _,inv_img = cv.threshold(img,250,255,cv.THRESH_BINARY_INV)
    img = cv.resize(inv_img,(28,28),interpolation = cv.INTER_AREA)
    all_images.append(img)
  return all_images

def data_set_mixer(X_train, X_test, y_train, y_test,num_classes=num_classes,Data_path=Data_path,labels=labels):
    train_images, test_images, train_labels, test_labels = Operators(Data_path , labels)
    X_train = np.append(X_train, train_images, axis=0)
    X_test = np.append(X_test, test_images, axis=0)
    y_train = np.append(y_train, train_labels)
    y_test = np.append(y_test, test_labels)
    Y_train = np_utils.to_categorical(y_train, num_classes)
    Y_test = np_utils.to_categorical(y_test, num_classes)
    return X_train, X_test, Y_train, Y_test
