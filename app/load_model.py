"""Load the model for future usage in the flask app"""
import keras

def load_model(path):
    loaded_model = keras.models.load_model(path)
    return loaded_model

if __name__=="__main__":
    is_loaded = load_model()
    print(type(is_loaded))
    fake_input = "./data/Mole_Data/SET_D/D95.BMP"
    p = CustomModel().preprocess_input(fake_input)
    pred = is_loaded.predict(p)
    print(pred)
    print(is_loaded.predict(fake_input))
    #D:\BECODE\Git\challenge-mole\app\model_save