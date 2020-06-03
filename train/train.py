
from config import num_classes,processed_path
from preprocessing.process import process_image
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.optimizers import Adam
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils, to_categorical
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, GlobalAveragePooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers
from tensorflow import keras


def make_model(num_classes=num_classes):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=(28,28,1)))
    model.add(Activation('relu'))
    BatchNormalization(axis=-1)
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    BatchNormalization(axis=-1)
    model.add(Conv2D(64,(3, 3)))
    model.add(Activation('relu'))
    BatchNormalization(axis=-1)
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Flatten())
    BatchNormalization()
    model.add(Dense(512))
    model.add(Activation('relu'))
    BatchNormalization()
    model.add(Dropout(0.2))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])
    return model

def generate_dataset(X_train, Y_train,X_test, Y_test):
    gen = ImageDataGenerator(rotation_range=8, width_shift_range=0.08, shear_range=0.3,height_shift_range=0.08, zoom_range=0.08)
    test_gen = ImageDataGenerator()
    test_gen = ImageDataGenerator()
    train_generator = gen.flow(X_train, Y_train, batch_size=64)
    test_generator = test_gen.flow(X_test, Y_test, batch_size=64)
    return train_generator,test_generator

def train_model(X_train, Y_train,X_test, Y_test,model_path,epochs=5,shuffle=True,num_classes=12):
    train_generator,test_generator = generate_dataset(X_train, Y_train,X_test, Y_test)
    Digit_Recognizer_model = make_model(num_classes)
    history = Digit_Recognizer_model.fit_generator(train_generator, steps_per_epoch=60000//64, epochs=epochs,validation_data=test_generator, validation_steps=10000//64,shuffle=shuffle)
    model.save_weights(model_path)
    print("Saved model to: ",model_path)
    return Digit_Recognizer_model

def load_model(model_path,num_classes=num_classes):
    Digit_Recognizer_model = make_model(num_classes)
    Digit_Recognizer_model.load_weights(model_path)
    return Digit_Recognizer_model
