import os
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
from flask import Flask, request, jsonify

# Disable GPU usage if CUDA drivers are not available
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Flask app
app = Flask(__name__)

# Debugging: Check if the model directory exists and list its contents
model_dir = "saved_model"
if not os.path.exists(model_dir):
    raise FileNotFoundError(f"Model directory '{model_dir}' does not exist.")
print(f"Contents of '{model_dir}': {os.listdir(model_dir)}")

# Load the TensorFlow SavedModel
model = tf.saved_model.load(model_dir)

# Load the labels file
labels_path = "labels.txt"
if not os.path.exists(labels_path):
    raise FileNotFoundError(f"Labels file '{labels_path}' does not exist.")
class_names = [line.strip() for line in open(labels_path, "r").readlines()]

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the image file from the request
        file = request.files.get('file')
        if file is None:
            return jsonify({'error': 'No file provided'}), 400

        # Preprocess the image
        image = Image.open(file).convert("RGB")
        size = (224, 224)
        image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
        image_array = np.asarray(image)
        normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
        data = np.expand_dims(normalized_image_array, axis=0)

        # Perform prediction
        infer = model.signatures["serving_default"]
        input_key = list(infer.structured_input_signature[1].keys())[0]
        output_key = list(infer.structured_outputs.keys())[0]
        prediction = infer(tf.constant(data))[output_key].numpy()

        # Get the predicted class and confidence score
        index = np.argmax(prediction)
        class_name = class_names[index]
        confidence_score = prediction[0][index]

        return jsonify({
            'class': class_name,
            'confidence_score': float(confidence_score)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
