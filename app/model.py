"""Custom Class using transfer learning : ResNet50"""
import keras
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg16 import VGG16
from keras.applications.inception_v3 import InceptionV3
from keras.applications.mobilenet import MobileNet
from keras.applications.densenet import DenseNet121
from keras.applications.xception import Xception

from keras.models import Model
from keras.layers import Dense,Dropout
from keras.layers import Flatten
import pandas as pd
import cv2
import numpy as np 
import os
from keras.preprocessing.image import array_to_img, img_to_array, load_img
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from .preprocess_train import ImageToNumpy

class CustomModel:
    """CustomModel based on ResNet50"""

    def __init__(self,X,y,model_name):
        self.X = X
        self.y = y
        self.model_name = model_name
        self.callbacks = self.set_call_back()
        self.X_train,self.X_test,self.y_train,self.y_test = train_test_split(self.X,self.y,test_size=0.2,random_state=42,shuffle=True)
    
    def initialize_model(self):
        """Model creation shaped for our images shapes

        Returns:
            [keras.model]: ResNet50, VGG16, InceptionV3,MobileNet, DenseNet121 or Xception
            depending on the model_name
        """
        model = eval(self.model_name)(include_top=False, input_shape=(64, 64, 3))

        flat1 = Flatten()(model.layers[-1].output)
        class1 = Dense(200, activation='relu')(flat1)
        output = Dense(3, activation='softmax')(class1)

        model = Model(inputs=model.inputs, outputs=output)
        return model
    
    def __get_call_back(self):
        return self.callbacks

    def set_call_back(self,monitor='val_accuracy',verbose=1,patience=2,min_delta=0.2):
        early_stop = EarlyStopping(monitor=monitor, verbose=verbose, patience=patience, min_delta=min_delta)
        checkpointer = ModelCheckpoint(filepath = "app/model_save_/{}".format(self.model_name), verbose = 2, save_best_only = True)
        self.callbacks = [checkpointer,early_stop]
        #self.callbacks = early_stop
        return self.callbacks

    def training_model(self,model,callbacks=True):
        es = None
        if callbacks:
            es = self.__get_call_back()
        
        model.compile(optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy'])
        
        history = model.fit(self.X_train, self.y_train,epochs=10,validation_data=(self.X_test, self.y_test),callbacks=[es])

        return history
    
    def save_model(self,model,save_name:str):
        model.save(save_name)
        return
    
    def load_model(self,save_name:str):
        try:
            reconstructed_model = keras.models.load_model(save_name)
            return reconstructed_model
        except FileNotFoundError:return None
    
    def save_exist(self,save_name:str)->bool:
        """Check if folder save_name exist

        Args:
            save_name (str): forlder name
        """
        pass
    
    def preprocess_input(self,path_input:str):
        """Prepare the image in the same fashion as the training set images"

        Args:
            path_input (str): path imput
        Returns:
            [np.array]:  array of shape(64,64,3)
        """
        p = ImageToNumpy(path_input)
        v = p.preprocess_input()
        return v


  
    

