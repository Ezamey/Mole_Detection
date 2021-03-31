from flask import Flask, render_template,flash,redirect,url_for,request,Response,send_file,make_response
from flask import current_app as app
import os
from .load_model import load_model
from .preprocess_train import ImageToNumpy
from mobileCam import mobileCam
from glob import glob

try:
    modelRes = load_model("./app/model_save_/ResNet50")
    print("ResNet50 loaded !")
    modelMobile = load_model("./app/model_save_/MobileNet")
    print("MobileNet loaded !")
except:
    print("Prob Error Path")

UPLOAD_FOLDER = 'app/static/image/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
myFeed = mobileCam()

@app.route('/detection',methods=['GET','POST'])
@app.route('/',methods=['GET','POST'])
def home():
    """
    Display the home page
    """
    #input_image = "IMAGE VIENT ICI"
    #predict_mobile = modelMobile.predict(p)
    #predict_res = modelRes.predict(p)   
    return render_template("detection.html")

@app.route('/uploader', methods = ['GET', 'POST'])
def upload_file():
   if request.method == 'POST':
      f = request.files['file']
      f.save(secure_filename(f.filename))

      preproc_img,path = ImageToNumpy().preprocess_single_input(f)
      prediction = modelRes.predict(preproc_img)
      
      return Response(img_path = path,prediction= prediction)

@app.route('/d',methods=['GET','POST'])
def detection():
    """
    Display the detection page  
    """
    return render_template("detection.html")

@app.route("/video_feed")
def video_feed():
	# return the response generated along with the specific media
	# type (mime type)
	return Response(myFeed.generate(),
		mimetype = "multipart/x-mixed-replace; boundary=frame")

@app.route("/close_feed",methods=['GET','POST'])
def close_feed():
    print('close')
	# return the response generated along with the specific media
	# type (mime type)
    return Response(myFeed.close_feed())



@app.route("/capture_feed",methods=['GET','POST'])
def capture_feed():
    print('capturing')
	# return the response generated along with the specific media
	# type (mime type)

    myFeed.capture_frame()
    processed_img,img_with_contours = myFeed.prediction()
    path = img_with_contours[4:]
    prediction = modelRes.predict(processed_img)
    response = Response(path)
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    response.headers['Cache-Control'] = 'public, max-age=0'
    # UPDATE WEBCAM WINDOW WITH PREDICTION IMG (draw_img)
    return response