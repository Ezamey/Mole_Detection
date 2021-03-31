import keras
from keras import layers, models
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import img_to_array, load_img
from keras.applications.mobilenet import MobileNet
from keras.models import Model
from keras.layers import Dense
from keras.layers import Flatten
from keras.utils import np_utils
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
from keras.callbacks import ModelCheckpoint
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
import matplotlib.pyplot as plt 
import cv2
from numpy import load
import numpy as np


## TO DO ##############################
# To test resize X 
# Train with datagen
# INTEGRATE FRAME OF CV TO HTML CHECK TUTORIAL 
# TRAIN DENSNET

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

"""
Import of features and target
"""
def load_data():
    y = load('./data/preprocessed/target.npy')
    X = load('./data/preprocessed/features.npy')
    X = X[2000:]
    y = y[2000:]
    print('Loaded data')
    # I do -1 on y to get 0,1,2 values or it will be an error later 
    y = y -1
    no_of_classes = len(np.unique(y))
    y = np_utils.to_categorical(y,no_of_classes)
    print('To categorical')
    return y,X



"""
Split data to train and test 
"""
def split_data(y,X):
    print('splitting')
    x_train, x_test, y_train, y_test = train_test_split(
       X, y, test_size=0.2, random_state=42)
    print('splited data')
    return x_train, x_test, y_train, y_test


def launch_model():
    model = MobileNet(include_top=False, input_shape=(387, 632, 3))
    model = Model(inputs=model.inputs, outputs=model.layers[-1].output)

    flat1 = Flatten()(model.layers[-1].output)
    class1 = Dense(50, activation='relu')(flat1)
    output = Dense(3, activation='softmax')(class1)

    model = Model(inputs=model.inputs, outputs=output)
    model.summary()

    model.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])
    print('Compiled!')
    return model


# def generate_data(x_train):
#     datagen = ImageDataGenerator(
#         featurewise_center=False,
#         featurewise_std_normalization=False,
#         rotation_range=180,
#         width_shift_range=0.2,
#         height_shift_range=0.2,
#         zoom_range= 0.1,
#         horizontal_flip=True,
#         vertical_flip= True)
#     datagen.fit(x_train)



def train_model(model,x_train,y_train,x_test,y_test):
    print(x_train.shape,y_train.shape,x_test.shape,y_test.shape)
    batch_size = 12 
    steps_per_epoch=len(x_train) / batch_size
    epochs = 10 
    checkpointer = ModelCheckpoint(filepath = './model/mobilenet_nomask.hdf5', verbose = 2, save_best_only = True)
    earlyStop = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
    history = model.fit(x_train,y_train,
            batch_size = batch_size,
            epochs=10,
            steps_per_epoch=steps_per_epoch,
            validation_data=(x_test, y_test),
            callbacks = [checkpointer,earlyStop],
            verbose=2, shuffle=True)
    print('model trained')
    return history


def evaluate_model(model,x_test,y_test):
    score = model.evaluate(x_test, y_test, verbose=0)
    print('\n', 'Test accuracy:', score[1])
    return score


def plot_model(history):
    # summarize history for accuracy  
    plt.figure(1)   
    plt.subplot(211)  
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])  
    plt.title('model accuracy')  
    plt.ylabel('accuracy')  
    plt.xlabel('epoch')  
    plt.legend(['train','test'], loc='upper left')  
    
    # summarize history for loss  
    plt.subplot(212)  
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])  
    plt.title('model loss')  
    plt.ylabel('loss')  
    plt.xlabel('epoch')  
    plt.legend(['train','test'], loc='upper left')  
    plt.show()
    return plt



def predict_image(path,model):
    file = ''
    new_image = []
    # TAKE THE IMG name(label)
    new_image.append(img_to_array(load_img(file, target_size=(100,100))))
    new_image = np.array(new_image).astype('float32')/255
    new_pred = model.predict(new_image)
    pred_idx = np.argmax(new_pred[0])
    return target_labels[pred_idx]



y,X = load_data()
x_train, x_test, y_train, y_test = split_data(y,X)
print(x_train.shape,y_train.shape,x_test.shape,y_test.shape)
model = launch_model()
history = train_model(model,x_train,y_train,x_test,y_test)
score = evaluate_model(model,x_test,y_test)
myplot = plot_model(history)

