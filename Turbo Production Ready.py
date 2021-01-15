#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from flask import Flask, request, render_template, url_for, jsonify, stream_with_context
from skimage.io import imread
from skimage.filters import threshold_otsu
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from skimage import measure
from skimage.measure import regionprops
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import localization
import pytesseract
import json
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

UPLOAD_FOLDER = '/tempsample'
app = Flask("__name__")
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/prediction', methods=['POST'])
def predict():
    txt = " 11"
    if request.method == 'POST':
        img = request.files['file']
        car_image = imread(img, as_grey=True)
        gray_car_image = car_image * 255
        threshold_value = threshold_otsu(gray_car_image)
        binary_car_image = gray_car_image > threshold_value
        label_image = measure.label(binary_car_image)
        plate_objects_cordinates = []
        plate_like_objects = []

        for region in regionprops(label_image):
            if region.area < 50:
                continue

            minRow, minCol, maxRow, maxCol = region.bbox
            plate_like_objects.append(binary_car_image[minRow:maxRow,
                                          minCol:maxCol])
            plate_objects_cordinates.append((minRow, minCol,
                                                      maxRow, maxCol))
            rectBorder = patches.Rectangle((minCol, minRow), maxCol-minCol, maxRow-minRow, edgecolor="red", linewidth=2, fill=False)
        
        ans = ""
        for item in plate_like_objects:
            text = pytesseract.image_to_string(item, lang='eng')
            if text:
                i = 0
                for i in range(len(text)):
                    if text[i].isalnum():
                        break
                text = text[i:]
                for i in range(len(text)-1,0,-1):
                    if text[i].isalnum():
                        break
                text = text[:i+1] 
                #print(text)
                #return flask.jsonify(text)
                ans += text
            
        return ans
    txt2 = "ttxt 2"
    return txt2
if __name__ == '__main__':
    app.run()


# In[ ]:




