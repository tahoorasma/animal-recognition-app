import tensorflow as tf
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)

MODEL_PATH = 'animal_model.h5'
model = tf.keras.models.load_model(MODEL_PATH)

animal_labels = {0: 'Cat', 1: 'Dog', 3: 'Wild'}

def preprocess_image(image):
    image = image.resize((224, 224))  
    image = np.array(image) / 255.0   
    image = np.expand_dims(image, axis=0)  
    return image

@app.route('/recognize', methods=['POST'])
def recognize_animal():
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400

    image_file = request.files['image']
    image = Image.open(image_file)

    processed_image = preprocess_image(image)

    predictions = model.predict(processed_image)
    predicted_label_index = np.argmax(predictions, axis=1)[0]
    predicted_animal = animal_labels.get(predicted_label_index, "Unknown animal")

    return jsonify({"animal": predicted_animal})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
