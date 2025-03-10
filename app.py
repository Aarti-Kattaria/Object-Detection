from flask import Flask, request, jsonify
from ultralytics import YOLO
import cv2
import numpy as np

app = Flask(__name__)

# Load your model (Make sure the path is correct)
model = YOLO("Object Detection\Object Detection\ppe.pt")  # Update path as needed

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    img = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(img, cv2.IMREAD_COLOR)

    # Run inference
    results = model(img)
    detections = []
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box
            conf = float(box.conf[0])  # Confidence
            cls = int(box.cls[0])  # Class index

            detections.append({
                "class": cls,
                "confidence": conf,
                "bbox": [x1, y1, x2, y2]
            })

    return jsonify({'detections': detections})

if __name__ == '__main__':
    app.run(debug=True)  # Run locally with debugging enabled
