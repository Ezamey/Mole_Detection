import cv2
from urllib.request import urlopen
import numpy as np
from glob import glob
from app.preprocess_train import ImageToNumpy
# You need to download ipWebcam on your smartphone then 
# => start server and copy the ip adress un varible stream
# note you need to be connected on the same network.

#stream = cv2.VideoCapture('http://192.168.1.26:8080/video')
# open the feed

class mobileCam:
    
    def __init__(self):
        self.vidcap = None
        self.count = len(glob("data/upload/*.jpg"))
        self.encodedImage = None
        self.live = True
        self.frame = None
        

    def generate(self):
        self.vidcap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        cv2.namedWindow("Window")
        while (self.live):
            # Read the stream feed and get image/frame
            success, frame = self.vidcap.read()
            # encode the frame in JPEG format
            flag, encodedImage = cv2.imencode(".jpg",frame)
            self.encodedImage = encodedImage
                        # yield the output frame in the byte format
            yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
                bytearray(encodedImage) + b'\r\n')
            
    def close_feed(self):
        self.live != self.live
        self.vidcap.release()
        cv2.destroyAllWindows()
        
    def capture_frame(self):
        img_name = "data/upload/opencv_frame_{}.jpg".format(self.count)
        self.encodedImage = cv2.imdecode(self.encodedImage,cv2.IMREAD_COLOR)
        self.frame = cv2.imwrite(img_name,self.encodedImage)
        self.count += 1
        return

    def prediction(self):
        last_img = len(glob("data/upload/*.jpg")) - 1 
        if len(glob("data/upload/*.jpg")) > 2 :
            last_image_saved = glob("data/upload/*.jpg")[last_img]
            print(last_image_saved)
        else:
            last_image_saved = glob("data/upload/*.jpg")[0]
            print(last_image_saved)
        print(len(glob("data/upload/*.jpg")))
        preproc_img = ImageToNumpy().preprocess_single_input(last_image_saved)
        return preproc_img


