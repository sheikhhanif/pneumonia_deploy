from __future__ import division, print_function

import base64
import json
from io import BytesIO
from keras import backend as K
import numpy as np
import requests
from flask import Flask, request, jsonify
from keras.applications import inception_v3
from keras.preprocessing import image
from keras.models import load_model
from keras.backend import clear_session
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
import math
# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer
# from flask_cors import CORS

app = Flask(__name__)

# this is custome loss functin
# focal loss 
def focal_loss(alpha=0.25,gamma=2.0):
    def focal_crossentropy(y_true, y_pred):
        bce = K.binary_crossentropy(y_true, y_pred)
        
        y_pred = K.clip(y_pred, K.epsilon(), 1.- K.epsilon())
        p_t = (y_true*y_pred) + ((1-y_true)*(1-y_pred))
        
        alpha_factor = 1
        modulating_factor = 1

        alpha_factor = y_true*alpha + ((1-alpha)*(1-y_true))
        modulating_factor = K.pow((1-p_t), gamma)

        # compute the final loss and return
        return K.mean(alpha_factor*modulating_factor*bce, axis=-1)
    return focal_crossentropy


# model = #laoding model
MODEL_PATH = 'models/model.hdf5'
model = load_model(MODEL_PATH, custom_objects={'FocalLoss': focal_loss, 'focal_crossentropy':focal_loss()})
model._make_predict_function()


def model_predict(img_path, model):
    # Decoding and pre-processing base64 image
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    pred = model.predict(x)
    return pred


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        # Make prediction
        predict = model_predict(file_path, model)
        prediction = ''
        #pred = predict['predictions']
        if predict[0][0] >= 0.5:
            prediction = 'The Patient has ' + str(math.ceil(predict[0][0]*100)) +'% chance of Pneumona'
        else:
            prediction = 'The Patient is Normal with ' + str(math.ceil(predict[0][0]*100)) + '%'
        # pred_class = preds.argmax(axis=-1)            # Simple argmax
           # ImageNet Decode
        #result = 'Penumonia rate', preds           # Convert to string
        return prediction
    return None


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')
