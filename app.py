from __future__ import division, print_function
import re
import os
import sys
import glob
import math
import json
import base64
import requests
import numpy as np
from io import BytesIO
import tensorflow as tf
from keras import backend as K
from keras.models import load_model
from gevent.pywsgi import WSGIServer
from keras.preprocessing import image
from keras.backend import clear_session
from werkzeug.utils import secure_filename
from keras.applications import inception_v3
from flask import Flask, redirect, url_for, request, render_template, jsonify


tf.logging.info('TensorFlow')
tf.logging.set_verbosity(tf.logging.ERROR)
tf.logging.info('TensorFlow')


app = Flask(__name__)

# model = #laoding model
MODEL_PATH = 'models/model.h5'
model = load_model(MODEL_PATH)
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
        if predict[0][0] >= 0.5:
            prediction = 'The Patient has ' + str(math.ceil(predict[0][0]*100)) +'% chance of Pneumona'
        else:
            prediction = 'The Patient is Normal with ' + str(math.ceil(predict[0][0]*100)) + '%'
        return prediction
    return None


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')
if __name__ == '__main__':
    app.run()