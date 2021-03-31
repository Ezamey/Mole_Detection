from keras.models import load_model

import logging
import numpy as np


class MoleDetectionManager():
    """
    This class represent a Model Manager which allows to
    a risk of melanoma from a given picture.
    """

    def __init__(self, model_path: str):
        """
        Initialize the ModelManager object by loading the model

        Parameters:
        -----------
        model_path <str>: String representing the path of the saved model
        """
        self.model = load_model(model_path)
        logging.info(f'Model loaded from <{model_path}>')

    def predict(self, img_path: str) -> list:
        """
        This function return the prediction of the melanoma risk
        for the given picture.

        Parameters:
        -----------
        img_path <str>: String representing the path of the picture.
                        This is this picture which will be use to compute
                        the returned predictions

        Return:
        -------
        A list of tuples each predictions realized.
        Each tuple contains these two elements:
            - the rectangular coordinates of the zone analysed
            - the predictions associated to this zone
        """

        logging.info(f'Start the predictions on <{img_path}>')

        predictions = []

        # Load the picture
        img = None  # img_to_array(img_path)

        # Search the ROI
        roi = self.get_regions_of_interest(img)

        for coord in roi:
            # Create an image from the ROI
            x = np.array()
            # Make prediction for the ROI detected
            pred = self.model.predict(x)
            predictions.append((coord, pred))

        return predictions

    def get_regions_of_interest(self, img_as_array: np.array) -> list:
        """
        This function return the ROI from a given picture.
        A ROI is a zone containing a mole.

        Parameters:
        -----------
        img_path <np.array>: Array representing the picture on which
                             the research of mole must be done

        Return:
        -------
        A list of the rectangular coordinates where a mole is detected
        """

        coordinates = []

        # Search after the ROI

        return coordinates
