import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load the model
model = tf.saved_model.load("saved_model")

# Load the labels
class_names = open("labels.txt", "r").readlines()

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    image = Image.open(file).convert("RGB")

    # Resize and preprocess the image
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array

    # Predict
    infer = model.signatures["serving_default"]
    prediction = infer(tf.constant(data))[infer.structured_outputs.keys()[0]]
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    return jsonify({
        'class': class_name[2:].strip(),
        'confidence_score': float(confidence_score)
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)