#!/usr/bin/env python
# coding: utf-8

# In[7]:


import flask 
import json
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import cv2
from PIL import Image
from flask_cors import CORS
from keras.preprocessing.image import ImageDataGenerator
from numpy import load
import pandas as pd
import io


app = flask.Flask(__name__)
model = None
data = None

def load_t():
    global model
    model = load_model('health_app_chestXray.h5')
    global data
    data = load('data.npy')
    
    
@app.route("/predict", methods=["POST"])
def predict():
    if flask.request.method == "POST":
        if flask.request.files.get("image"):
            # read the image in PIL format
            image = flask.request.files["image"].read()
            image = Image.open(io.BytesIO(image))
            
        image.save('sample_test_1.png')
        
        d = {'Image' : ['sample_test_1.png'],
        'Cardiomegaly':[0], 
          'Emphysema':[0], 
          'Effusion':[0], 
          'Hernia':[0], 
          'Infiltration':[0], 
          'Mass':[0], 
          'Nodule':[0], 
          'Atelectasis':[0],
          'Pneumothorax':[0],
          'Pleural_Thickening':[0], 
          'Pneumonia':[0], 
          'Fibrosis':[0], 
          'Edema':[0], 
          'Consolidation':[0]}
        
        test_df = pd.DataFrame(d, columns = ['Image','Cardiomegaly', 
          'Emphysema', 
          'Effusion', 
          'Hernia', 
          'Infiltration', 
          'Mass', 
          'Nodule', 
          'Atelectasis',
          'Pneumothorax',
          'Pleural_Thickening', 
          'Pneumonia', 
          'Fibrosis', 
          'Edema', 
          'Consolidation'])
        
        global data
        image_generator = ImageDataGenerator(
        featurewise_center=True,
        featurewise_std_normalization= True)
        image_generator.fit(data)
        
        labels = ['Cardiomegaly', 
          'Emphysema', 
          'Effusion', 
          'Hernia', 
          'Infiltration', 
          'Mass', 
          'Nodule', 
          'Atelectasis',
          'Pneumothorax',
          'Pleural_Thickening', 
          'Pneumonia', 
          'Fibrosis', 
          'Edema', 
          'Consolidation']
        
        test_generator = image_generator.flow_from_dataframe(
            dataframe=test_df,
            directory='projects/testing/',
            x_col="Image",
            y_col=labels,
            class_mode="raw",
            batch_size=16,
            shuffle=False,
            seed=1,
            target_size=(320,320))
        
        global model
        pred = model.predict(test_generator, steps = len(test_generator))
        
        ref = {
                0:'Cardiomegaly', 
          1:'Emphysema', 
          2:'Effusion', 
          3:'Hernia', 
          4:'Infiltration', 
          5:'Mass', 
          6:'Nodule', 
          7:'Atelectasis',
          8:'Pneumothorax',
          9:'Pleural_Thickening', 
          10:'Pneumonia', 
          11:'Fibrosis', 
          12:'Edema', 
          13:'Consolidation'
        }
        
        prediction = ref[pred]
        res_data = {'pred': prediction}
        os.remove('sample_1.png')
        return flask.jsonify(pred)
    
if __name__ == "__main__":
    load_t()
    from waitress import serve
    serve(app, host="0.0.0.0", port=8081)    


# In[ ]:




