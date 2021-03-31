"""Entry of our app"""
import pandas as pd
import numpy as np
from tensorflow.keras.utils import to_categorical
import os

from app.preprocess_train import ImageToNumpy
from app.model import CustomModel
from app import init_app


if __name__ == "__main__":

    # Possible model ResNet50, VGG16, InceptionV3, MobileNet, DenseNet121 or Xception
    modelMobile = 'MobileNet'
    modelResNet = "ResNet50"
    model_names = [modelMobile,modelResNet]

    #Checking if save exist model part
    CUR_DIR = os.getcwd()
    Mobile_exist = os.path.exists(os.path.join(CUR_DIR,"app/model_save_/{}".format(modelMobile)))
    ResNet50_exist = os.path.exists(os.path.join(CUR_DIR,"app/model_save_/{}".format(modelResNet)))
    existing_model_booleans = [Mobile_exist,ResNet50_exist]

    #list of tuples (ModelsNames,Boolean)
    model_exist_and_names = zip(model_names,existing_model_booleans)

    print(model_exist_and_names)
    for model_in in model_exist_and_names:
        print(model_in)
        if not model_in[1]:
            print("No saved model found ! Training and saving it")
            #Dataframe Operation
            #PATH = "./data/preprocessed/df_mole_merged.csv"
            #df = pd.read_csv(PATH)
            #^ Not needed anymore since our data are in the features and target npy files

            y = np.load("./data/preprocessed/target.npy")
            y = y-1
            y = to_categorical(y,num_classes=3,dtype='float32')

            #converting train images to np.array and saving it in: 
            X = ImageToNumpy().dataset_builder()

            #Init,train and save
            custom_model = CustomModel(X,y,model_in[0])
            model = custom_model.initialize_model()
            history = custom_model.training_model(model=model)
            custom_model.save_model(model,"app/model_save_/{}".format(model_in[0]))

    #pred
    if all(existing_model_booleans):
        print("Saved models found !")
 
    app = init_app().run(debug=False)
