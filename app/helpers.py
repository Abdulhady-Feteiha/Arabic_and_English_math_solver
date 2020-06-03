from train.train import load_model,train_model
from config import BASE_PATH, processed_path
from data_loader.load_dataset import Arabic,English
from preprocessing.process import process_image, process_dataset
from preprocessing.segment import segment
import glob
import numpy as np
import os

def build_recognizer(BASE_PATH=BASE_PATH):

    print("Loading Arabic dataset")
    X_train, X_test, Y_train, Y_test  = Arabic()
    X_train, X_test = process_dataset(X_train, X_test)
    ar_model_path = os.path.join(BASE_PATH, r"train/ar_model.h5")
    print("Training Arabic model")
    ar_model = train_model(X_train, Y_train,X_test, Y_test,ar_model_path)

    print("Loading English dataset")
    X_train, X_test, Y_train, Y_test  = English()
    X_train, X_test = process_dataset(X_train, X_test)
    en_model_path = os.path.join(BASE_PATH, r"train/en_model.h5")
    print("Training Arabic model")
    en_model = train_model(X_train, Y_train,X_test, Y_test,en_model_path)
    return ar_model, en_model

def calculate(img_path,model_path,digital,processed_path=processed_path):
    segment(img_path,digital)
    Digit_Recognizer_model = load_model(model_path)
    chars = predict(Digit_Recognizer_model,processed_path,digital)
    print(chars)
    result = operation(chars)
    return result

def predict(Digit_Recognizer_model,processed_path,digital):
    imgs = sorted(glob.glob(processed_path+"/*.png"))
    predicted = []
    for img in imgs:
      test_img = process_image(img,digital)
      pred = Digit_Recognizer_model.predict(test_img)
      pred = pred[0]
      pred_digit = np.where(pred == np.amax(pred))[0][0]
      predicted.append(pred_digit)
    return predicted

def get_number(number):
    number_1=0
    for i in range(len(number)):
        number_1=number_1+number[i]*10**(len(number)-i-1)
    return number_1

def operation(test):
    operators_index=[index for index,value in enumerate(test) if value > 9]
    numbers=[]
    for i in range(len(operators_index)+1):
      if i==0:
        number=test[:operators_index[i]]
      elif i==len(operators_index):
        number=test[operators_index[i-1]+1::]
      else:
        number=test[operators_index[i-1]+1:operators_index[i]]
      numbers.append(get_number(number))
    # print(numbers)

    result=numbers[0]
    for i in range(len(operators_index)):
      if test[operators_index[i]]==10:
        result=result+numbers[i+1]
      elif test[operators_index[i]]==11:
        result=result-numbers[i+1]

    return result
