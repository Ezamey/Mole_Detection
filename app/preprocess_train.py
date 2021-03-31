"""script responsible of converting the train set into a numpy array after detecting ROI and check if the .npy file exist for futur use"""

import pandas as pd
import cv2
import numpy as np 
import os
import datetime
from keras.preprocessing.image import array_to_img, img_to_array, load_img

class ImageToNumpy:
    """Open image with opencv2->find ROI->Crop img*n_time  -> save it in .np file
    """

    def __init__(self,df_object=None):
        self.df_object = df_object
        is_frame = isinstance(self.df_object,pd.core.frame.DataFrame)
        if is_frame:
            self.files_path = df_object['file_path']
            self.counter = 0
        if not is_frame:
            self.counter = 0
            print("ImageToNumpy initialized without pd.core.frame.DataFrame object")


    def __create_contour(self,img2):
        """Apply a mask to an image. Finding the ROI

        Args:
            file (str): image path

        Returns:
            [np.array]: matrice array of the masked image
        """
        if isinstance(img2,str):
            img2 = cv2.imread(img2)
        img = img2.copy()
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        #this code is used to find average color
        avg_color_per_row = np.average(img, axis=0)
        avg_color = np.average(avg_color_per_row, axis=0)
    
        img_blur = cv2.GaussianBlur(img, (9,9), 0)
        _, img_binary = cv2.threshold(img_blur,(avg_color-(avg_color/2)), 255, cv2.THRESH_BINARY_INV)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9,9))
        dilation = cv2.erode(img_binary, kernel, iterations= 2)
        closed = cv2.morphologyEx(dilation,cv2.MORPH_CLOSE, kernel)

        cnts = cv2.findContours(img_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]

        final = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        areaArray = []
        count = 1
        for i,c in enumerate(cnts):
            x,y,w,h = cv2.boundingRect(c)
            #cv2.rectangle(final, (x - 40,y - 40), (x+w + 40, y+h + 40), (0,0,0), -1)
            area = cv2.contourArea(c)
            areaArray.append(area)
        #_, img_binary = cv2.threshold(final,100, 255, cv2.THRESH_BINARY_INV)

        #image_masked = cv2.bitwise_and(img2, img_binary)

        self.counter +=1
        print("{} images masquées".format(self.counter))
        sorteddata = sorted(zip(areaArray, cnts), key=lambda x: x[0], reverse=True)

        return sorteddata

    def __find_index(self,sorted_data):
        """Find the index of the biggest rectange in zipped list

        Args:
            sorted_data ([list of tuples]): list of(areaArray,contour)

        Returns:
            contours for the biggest area.
        """
        biggest = max([x[0] for x in sorted_data])
        for data in sorted_data:
            if data[0]==biggest:
                return data[1]
    
    def __cropping(self,img,contour,padding=5):
        x, y, w, h = cv2.boundingRect(contour)
        padding_y = 5
        padding_x = 5
        padding_w = 5
        padding_h = 5
        cond_1 = y-padding<0
        cond_2 = x-padding<0
        cond_3 = x+w > img.shape[0]
        cond_4 = y+h > img.shape[1]
        if cond_1:           
            padding_y = 0
        if cond_2:
            padding_x = 0
        if cond_3:
            padding_w = 0
        if cond_4:
            padding_h = 0
            
        ret_img = img[y-padding_y:y+h+padding_h, x-padding_x:x+w+padding_w]
        return ret_img
    
    def __resize_image(self,img):
        """resize image in format 64*64 without keeping ratios

        Args:
            img = opencv.img

        Returns:
            img = resized opencv.img
        """
        width = 64
        height = 64
        dim = (width, height)
        # resize image
        resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
        return resized

    def dataset_builder(self):
        """Build features set by applying a mask to every image path in the dataframe.
        Then convert them into a numpy array and save it in .npy file

        Returns:
            [np.array]: [np.array representing all masked images]
        """

        saving_path = './data/Mole_Data/features.npy'
        #To change if you have a personnal .npy file you want to train on
        np_file_exist = os.path.exists('./data/Mole_Data/features.npy') 

        if not np_file_exist:
            images_as_array=[]
            img_array = np.load('./data/preprocessed/features.npy')
            for img_as_np in img_array:

                path = np.array(img_as_np*255).astype('uint8')
                #contour et trouver le plus gros contour
                contour = self.__create_contour(path)
                largest_contour = self.__find_index(contour)
                #cropping
                image = self.__cropping(path,largest_contour)
                #reshape
                resized = self.__resize_image(image)
                #add reshape to images_as_array
                images_as_array.append(img_to_array(resized))
            
            print("Array conversion")
            arr = np.array(images_as_array).astype('float32')/255
            print("Array conversion done. Saving in {}".format(saving_path))
            np.save(saving_path,images_as_array)
            return arr
        print('{} already exist'.format(saving_path))
        print("Loading...")
        return np.load(saving_path,allow_pickle=True)
    
    def preprocess_input(self):
        """transform the input image in the same fashion of the train set

        Returns:
            [np.array]: np.array of shape (64,64,3)
        """
        path = self.df_object
        contour = self.__create_contour(path)
        #trouver le plus gros contour
        largest_contour = self.__find_index(contour)
        #image
        image = self.__cropping(cv2.imread(path),largest_contour)              
        #todo : reshape
        resized = self.__resize_image(image)
        #add reshape to images_as_array
        #conversion
        r_to_array = img_to_array(resized)

        #expand dims for keras model input
        final = np.expand_dims(np.array(r_to_array).astype('float32')/255,0)
        print("Img conversion done.")
        return final

    def preprocess_single_input(self,path):
        """transform the input image in the same fashion of the train set

        Returns:
            [np.array]: np.array of shape (64,64,3)
        """
        img = cv2.imread(path)
        contour = self.__create_contour(img)
        #trouver le plus gros contour
        largest_contour = self.__find_index(contour)
        #image
        image = self.__cropping(img,largest_contour)              
        #todo : reshape
        resized = self.__resize_image(image)
        #add reshape to images_as_array
        #conversion
        r_to_array = img_to_array(resized)

        #expand dims for keras model input
        final = np.expand_dims(np.array(r_to_array).astype('float32')/255,0)

        now = datetime.datetime.now()
        img_with_contours_path = 'app/static/image/tmp/img_with_contours'+ now.strftime('%H%M%S') +'.jpg'
        ## If file exists, delete it ##
        if os.path.isfile(img_with_contours_path):
            os.remove(img_with_contours_path)
            print('deleted file')
        else:   ## Show an error ##
            print("Error: %s file not found" % img_with_contours_path)
        img_with_contours = cv2.drawContours(img, largest_contour, -1, (0,255,0), 3)
        img_with_contours = cv2.imwrite(img_with_contours_path,img_with_contours)

        # text = "{}: {:.2f}%".format(age, age_confidence * 100)
        # cv.putText(image, text, (start_x, y), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        print("Img conversion done.")
        return final,img_with_contours_path


