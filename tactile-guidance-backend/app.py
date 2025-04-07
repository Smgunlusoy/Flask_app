import sys
import os
import cv2
import json
import numpy as np
import torch
import time
from flask import Flask, request, jsonify, Response
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Correct path configuration to find the aibox module
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

# Verify if aibox exists
aibox_path = os.path.join(project_root, 'aibox')
if not os.path.isdir(aibox_path):
    logging.error(f"Error: aibox directory not found at {aibox_path}")
    sys.exit(1)

try:
    from aibox.bracelet import connect_belt, BraceletController
    logging.info("Successfully imported aibox modules")
except ImportError as e:
    logging.error(f"Failed to import from aibox.bracelet: {e}")
    sys.exit(1)

# Flask app initialization
app = Flask(__name__)

# Load YOLOv5 model
model_yolo = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Load hand detection model
hand_model_path = os.path.join(aibox_path, 'hand.pt')
if not os.path.isfile(hand_model_path):
    logging.error(f"Error: hand.pt file not found at {hand_model_path}")
    sys.exit(1)

model_hand = torch.hub.load('ultralytics/yolov5', 'custom', path=hand_model_path)

# COCO labels
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

# Camera detection
camera_index = next((i for i in range(10) if cv2.VideoCapture(i).isOpened()), 0)
video_camera = None
bracelet_controller = None
belt_controller = None

@app.route('/')
def home():
    return "✅ Tactile Guidance Flask Backend is Live!"

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/detected_objects', methods=['GET'])
def get_detected_objects():
    global video_camera
    if video_camera is None:
        video_camera = cv2.VideoCapture(camera_index if camera_index is not None else 0)

    if not video_camera.isOpened():
        return jsonify([])

    ret, frame = video_camera.read()
    if not ret:
        return jsonify([])

    results = model_yolo(frame)
    detections = results.xyxy[0].cpu().numpy()

    found = set()
    for det in detections:
        if len(det) < 6:
            continue
        _, _, _, _, conf, cls = det
        cls = int(cls)
        if 0 <= cls < len(coco_labels):
            label = coco_labels[cls]
            found.add(label)

    return jsonify(list(found) if found else ["No recognizable objects detected."])

@app.route('/activate_bracelet', methods=['POST'])
def activate_bracelet():
    data = request.get_json()
    object_name = data.get('object_name')
    if not object_name:
        return jsonify({"error": "Object name not provided"}), 400

    logging.info(f"🔧 Received request to guide toward: {object_name}")
    try:
        guide_bracelet_to_object(object_name)
        return jsonify({"message": f"Bracelet guiding started for: {object_name}"}), 200
    except Exception as e:
        logging.error(f"❌ Error: {e}")
        return jsonify({"error": str(e)}), 500

def generate_frames():
    global video_camera
    video_camera = cv2.VideoCapture(camera_index)
    while True:
        success, frame = video_camera.read()
        if not success:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results_yolo = model_yolo(frame)
        detections_yolo = results_yolo.xyxy[0].cpu().numpy()

        for det in detections_yolo:
            if len(det) < 6:
                continue
            x1, y1, x2, y2, conf, cls = det
            cls = int(cls)
            if cls < 0 or cls >= len(coco_labels):
                continue
            label = coco_labels[cls]
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
            cv2.putText(frame, label, (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        hand_results = model_hand(frame)
        detections_hand = hand_results.xyxy[0].cpu().numpy()

        for det in detections_hand:
            if len(det) < 6:
                continue
            x1, y1, x2, y2, conf, cls = det
            cls = int(cls)
            label = "hand"
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, label, (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def guide_bracelet_to_object(target_cls):
    global bracelet_controller, belt_controller

    if bracelet_controller is None:
        bracelet_controller = BraceletController()
        if hasattr(bracelet_controller, 'init'):
            bracelet_controller.init(vibration_intensities={
                'bottom': 50, 'top': 50, 'left': 50, 'right': 50
            })

    if not belt_controller:
        connected, belt_controller = connect_belt()
        if not connected:
            raise Exception("Could not connect to bracelet")

    cap = cv2.VideoCapture(camera_index)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        yolo_results = model_yolo(frame)
        detections_yolo = yolo_results.xyxy[0].cpu().numpy()

        bboxes = []
        for i, det in enumerate(detections_yolo):
            if len(det) < 6:
                continue
            x1, y1, x2, y2, conf, cls = det
            cls = int(cls)
            if cls < 0 or cls >= len(coco_labels):
                continue
            label = coco_labels[cls]
            bboxes.append([x1, y1, x2 - x1, y2 - y1, i, label, conf])

        hand_results = model_hand(frame)
        detections_hand = hand_results.xyxy[0].cpu().numpy()

        for det in detections_hand:
            if len(det) < 6:
                continue
            x1, y1, x2, y2, conf, cls = det
            cls = int(cls)
            label = "hand"
            bboxes.append([x1, y1, x2 - x1, y2 - y1, 999, label, 0.99])

        # Bu kısımda bboxes listenizi kontrol edin ve hedef nesneyi bulun
        target_bbox = None
        for bbox in bboxes:
            if bbox[5] == target_cls:
                target_bbox = bbox
                break

        if target_bbox:
            bracelet_controller.navigate_hand(
                belt_controller,
                [target_bbox],
                target_cls,
                hand_clss=["hand"],
                depth_img=None
            )

        time.sleep(0.5)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000, debug=True)