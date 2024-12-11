import os
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
from flask import Flask, request, jsonify

app = Flask(__name__)

# Disable GPU usage
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Debugging: Check if the model directory exists and list its contents
model_dir = "saved_model"
if not os.path.exists(model_dir):
    raise FileNotFoundError(f"Model directory '{model_dir}' does not exist.")
print(f"Contents of '{model_dir}': {os.listdir(model_dir)}")

# Load the model
model = tf.saved_model.load(model_dir)

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
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)