from __future__ import division, print_function
# coding=utf-8
import os
import numpy as np
from PIL import Image
# Keras
from keras.models import load_model
from keras.preprocessing import image
# Flask utils
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename

# Define a flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-goes-here'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

MODEL_PATH = 'Model.hdf5'

# Load your trained model
print(" ** Model Loading **")
model = load_model(MODEL_PATH)
print(" ** Model Loaded **")

def model_predict(img_path, model):
    img = Image.open(img_path).resize((224, 224))

    # Preprocessing the image
    x = np.asarray(img)
    # x = np.true_divide(x, 255)
    x = np.expand_dims(x, axis=0)

    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
    x = x/255

    preds = model.predict(x)
    d = preds.flatten()
    j = d.max()
    li=['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy', 'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy']
    for index,item in enumerate(d):
        if item == j:
            class_name = li[index].split('___')
    return class_name

@app.route('/')
def index():
    # Main page
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':

        # Check if the request contains a file
        if 'file' not in request.files:
            return "No file found in the request"
        f = request.files['file']
        
        # Check if the file is empty
        if f.filename == '':
            return "Empty file name"
        
        # Check if the file extension is allowed
        allowed_extensions = {'png', 'jpg', 'jpeg', 'gif'}
        if not '.' in f.filename or f.filename.rsplit('.', 1)[1].lower() not in allowed_extensions:
            return "Invalid file extension"
        
        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        try:
            class_name = model_predict(file_path, model)
            result = f"Predicted Crop: {class_name[0]}  Predicted Disease: {class_name[1].title().replace('_', ' ')}"
            return result
        except Exception as e:
            return "Prediction failed: " + str(e)
    return "No file uploaded"

if __name__ == '__main__':
    app.run(debug=True)
