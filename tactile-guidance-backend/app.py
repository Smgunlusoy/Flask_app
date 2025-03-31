import sys
import os
import cv2
import json
import numpy as np
import torch
import logging
from flask import Flask, request, jsonify, Response
from aibox.bracelet import connect_belt, BraceletController

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Add the parent directory of aibox to PYTHONPATH
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
aibox_path = os.path.join(project_root, 'aibox')
sys.path.append(aibox_path)  

print(f'aibox_path: {aibox_path}') 

# Verify if aibox exists
if not os.path.isdir(aibox_path):
    logging.error(f"The aibox directory does not exist at {aibox_path}")
    sys.exit(1)

try:

    from aibox.bracelet import connect_belt, BraceletController
    logging.info("Successfully imported aibox modules")
except ImportError as e:
    logging.error(f"Error importing aibox modules: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

app = Flask(__name__)

# Load the YOLO model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Define COCO labels
coco_labels = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
    "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop",
    "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]

# Detect the first available camera
camera_index = None
for i in range(10):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        camera_index = i
        cap.release()
        logging.info(f"Camera found: {camera_index}")
        break

if camera_index is None:
    logging.error("No camera found.")

@app.route('/')
def home():
    return "Welcome to the Tactile Guidance System!"

def generate_frames():
    camera = cv2.VideoCapture(camera_index if camera_index is not None else 0)
    if not camera.isOpened():
        logging.error("Could not open camera.")
        return

    while True:
        success, frame = camera.read()
        if not success:
            logging.error("Failed to capture frame.")
            break
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = model(frame_rgb)
        detections = results.xyxy[0].numpy()

        for det in detections:
            x1, y1, x2, y2, conf, cls = det
            label = f'{coco_labels[int(cls)]}, Score: {conf:.2f}'
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
            cv2.putText(frame, label, (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/detect_objects', methods=['POST'])
def detect_objects():
    try:
        webcam = cv2.VideoCapture(camera_index if camera_index is not None else 0)
        if not webcam.isOpened():
            return jsonify({"error": "Error opening OpenCV VideoCapture"}), 500
        
        ret, frame = webcam.read()
        if not ret:
            return jsonify({"error": "Error reading frame from camera"}), 500
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = model(frame_rgb)
        detections = results.xyxy[0].numpy()

        detected_objects = []
        for det in detections:
            x1, y1, x2, y2, conf, cls = det
            detected_objects.append({
                "class": coco_labels[int(cls)],
                "score": float(conf),
                "box": [float(x1), float(y1), float(x2), float(y2)]
            })
        
        webcam.release()
        return jsonify({"message": "Object detection completed", "objects": detected_objects}), 200
    except Exception as e:
        logging.error(f"Error in detect_objects: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/activate_bracelet', methods=['POST'])
def activate_bracelet():
    data = request.get_json()
    object_name = data.get('object_name')
    if not object_name:
        return jsonify({"error": "Object name not provided"}), 400
    try:
        guide_bracelet_to_object(object_name)
        return jsonify({"message": "Bracelet guiding started for object: " + object_name}), 200
    except Exception as e:
        logging.error(f"Error activating bracelet: {str(e)}")
        return jsonify({"error": str(e)}), 500

# Initialize the bracelet controller only once
bracelet_controller = None

def guide_bracelet_to_object(object_name):
    global bracelet_controller
    if bracelet_controller is None:
        bracelet_controller = BraceletController(vibration_intensities={
            'bottom': 50, 'top': 50, 'left': 50, 'right': 50
        })

    connected, belt_controller = connect_belt()
    if connected:
        bboxes = []  # You would need actual bounding boxes for this to work appropriately
        target_cls = object_name  # Assuming object_name is your target class
        hand_clss = ["hand_class_id1", "hand_class_id2"]  # Update with actual class IDs
        depth_img = None  # Handle accordingly if you have depth data
        bracelet_controller.navigate_hand(belt_controller, bboxes, target_cls, hand_clss, depth_img)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000, debug=True, use_reloader=False)