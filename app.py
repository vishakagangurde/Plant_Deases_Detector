from flask import Flask, request, render_template, redirect, url_for
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'  # Folder to store uploaded images
model = load_model('C:/Users/visha/OneDrive/Desktop/PYTHON/C__Plant-Disease-Detection_Model_plant_disease_model.keras')

all_labels = [
    'Potato___Early_blight', 
    'Potato___Late_blight', 
    'Tomato_Leaf_Mold', 
    'Tomato_Spider_mites_Two_spotted_spider_mite', 
    'Potato___healthy', 
    'Tomato_Late_blight', 
    'Tomato_healthy', 
    'Tomato_Septoria_leaf_spot'
]

@app.route('/', methods=['GET', 'POST'])
def index():
    uploaded_image = None
    prediction = ''
    
    if request.method == 'POST':
        file = request.files['file']
        if file:
            # Save the uploaded file
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)
            uploaded_image = file.filename

            # Process the image
            image = Image.open(file_path)
            image = image.resize((128, 128))
            image = np.expand_dims(image, axis=0)
            image = np.array(image) / 255.0

            # Make prediction
            pred = model.predict(image)
            prediction = all_labels[np.argmax(pred)]

    return render_template('index.html', uploaded_image=uploaded_image, prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
