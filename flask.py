# -*- coding: utf-8 -*-
"""flask.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1W9laMXX7ASgQ0G5Zdx_kPmqzdi1I0kwz
"""

# prompt: connect to google drive

from google.colab import drive
drive.mount('/content/drive')

!pkill -f app.py

!fuser -k 5000/tcp

!pip install flask flask-cors ultralytics

!pip install flask-cors

# Commented out IPython magic to ensure Python compatibility.
# %%writefile app.py
# from flask import Flask, request, jsonify
# from flask_cors import CORS
# from ultralytics import YOLO
# from PIL import Image
# import io
# 
# # Initialize Flask
# app = Flask(__name__)
# CORS(app)
# 
# # Load YOLO model
# model = YOLO("/content/drive/MyDrive/DataCrumbs/Object Detection/Object Detection/ppe.pt")  # Update with your model path
# 
# @app.route("/detect", methods=["POST"])
# def detect():
#     try:
#         if "image" not in request.files:
#             return jsonify({"error": "No image uploaded"}), 400
# 
#         # Read the uploaded image
#         file = request.files["image"]
#         image = Image.open(file.stream)
# 
#         # Run detection
#         results = model(image)
# 
#         # Prepare the response
#         detections = []
#         for result in results:
#             for box in result.boxes:
#                 bbox = box.xyxy[0].tolist()  # Bounding box coordinates [x1, y1, x2, y2]
#                 score = float(box.conf[0])   # Confidence score
#                 label = int(box.cls[0])      # Class label
#                 detections.append({"bbox": bbox, "score": score, "label": label})
# 
#         return jsonify({"detections": detections})
# 
#     except Exception as e:
#         print("Error:", str(e))
#         return jsonify({"error": str(e)}), 500
# 
# if __name__ == "__main__":
#     app.run(host="0.0.0.0", port=5000, debug=True)
#

"""%%writefile app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO
from PIL import Image
import io

# Initialize Flask
app = Flask(__name__)
CORS(app)

# Load YOLO model
model = YOLO("/content/drive/MyDrive/DataCrumbs/Object Detection/Object Detection/ppe.pt")  # Update with your model path

@app.route("/detect", methods=["POST"])
def detect():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    # Read the uploaded image
    file = request.files["image"]
    image = Image.open(file.stream)

    # Run detection
    results = model(image)

    # Prepare the response
    detections = []
    for result in results[0].boxes:
        bbox = result.xyxy[0].tolist()  # Bounding box in [x1, y1, x2, y2]
        score = result.conf.tolist()    # Confidence score
        label = result.cls.tolist()     # Class label
        detections.append({"bbox": bbox, "score": score, "label": label})

    return jsonify({"detections": detections})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
"""

!pip install pyngrok

!ngrok config add-authtoken 2tw2jGQpxYe5p3nwNCqNJxKZdMI_6hVrStkY4ApaaPR9vMqLf

!nohup ngrok http 5000 --log=stdout &

!ngrok config add-authtoken 2tw2jGQpxYe5p3nwNCqNJxKZdMI_6hVrStkY4ApaaPR9vMqLf

!nohup ngrok http 5000 &

!ps aux | grep ngrok

!curl -s http://localhost:4040/api/tunnels | grep -o '"public_url":"[^"]*' | cut -d'"' -f4

import requests

# Open the image file
image_path = "/content/drive/MyDrive/DataCrumbs/Object Detection/Object Detection/bus.jpg"  # Update with your actual image path
files = {"image": open(image_path, "rb")}

# Send the request
response = requests.post("https://8732-34-86-243-101.ngrok-free.app/detect", files=files)

# Debugging: Print raw response first
print("Status Code:", response.status_code)
print("Raw Response:", response.text)  # This will show if there's an error page



!ps aux | grep app.py

!nohup python3 app.py --host=0.0.0.0 --port=5000 &

!ps aux | grep app.py

!pkill -9 ngrok
!pkill -9 python3

!nohup python3 app.py --host=0.0.0.0 --port=5000 &

!ps aux | grep app.py

!nohup ngrok http 5000 --log=stdout &

!curl -s http://localhost:4040/api/tunnels