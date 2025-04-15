import os
import sys
import cv2
import time
import torch
import logging
import numpy as np
from flask import Flask, request, jsonify, Response
from ultralytics import YOLO
from flask_cors import CORS
from pybelt.belt_controller import (
    BeltConnectionState,
    BeltController,
    BeltControllerDelegate,
    BeltMode,
    BeltOrientationType,
    BeltVibrationPattern
)

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Ensure root path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

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

app = Flask(__name__)
CORS(app)

# YOLO Models
try:
    model_yolo = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    logging.info("✅ YOLOv5 model loaded.")
except Exception as e:
    logging.error(f"❌ YOLOv5 model load failed: {e}")
    sys.exit(1)

try:
    model_hand = YOLO("hand_yolov8s.pt")
    logging.info("✅ YOLOv8 hand model loaded.")
except Exception as e:
    logging.error(f"❌ YOLOv8 model load failed: {e}")
    sys.exit(1)

# Globals
coco_labels = model_yolo.names
camera_index = next((i for i in range(10) if cv2.VideoCapture(i).isOpened()), 0)
video_camera = None
bracelet_controller = None
belt_controller = None

class Delegate(BeltControllerDelegate):
    def on_belt_connection_state_changed(self, state):
        logging.info(f"Belt connection state changed: {state}")

    def on_belt_button_pressed(self, press_count):
        logging.info(f"Belt button pressed {press_count} times")

@app.route("/")
def home():
    return "✅ Tactile Guidance Flask Backend is Live!"

@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/detected_objects", methods=["GET"])
def get_detected_objects():
    global video_camera
    if video_camera is None:
        video_camera = cv2.VideoCapture(camera_index)
    if not video_camera.isOpened():
        return jsonify([])

    ret, frame = video_camera.read()
    if not ret:
        return jsonify([])

    found = detect_objects(frame)
    return jsonify(list(found))

@app.route("/activate_bracelet", methods=["POST"])
def activate_bracelet():
    data = request.get_json()
    object_name = data.get("object_name", "").lower().strip()

    if not object_name:
        return jsonify({"error": "Missing object_name"}), 400

    try:
        guide_bracelet_to_object(object_name)
        return jsonify({"message": f"Guidance started for {object_name}"}), 200
    except Exception as e:
        logging.error(f"Guidance error: {e}")
        return jsonify({"error": str(e)}), 500

def detect_objects(frame):
    results = model_yolo(frame)
    detections = results.xyxy[0].cpu().numpy()
    found = set()

    for det in detections:
        if len(det) < 6:
            continue
        x1, y1, x2, y2, conf, cls = det
        label = coco_labels[int(cls)]
        if conf > 0.5 and label != "person":
            found.add(label)

    hands = model_hand(frame)
    if hands and hands[0].boxes is not None:
        for conf in hands[0].boxes.conf.cpu().numpy():
            if conf > 0.3:
                found.add("hand")
                break
    return found

def generate_frames():
    cap = cv2.VideoCapture(camera_index)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model_yolo(frame)
        detections = results.xyxy[0].cpu().numpy()

        for det in detections:
            if len(det) < 6:
                continue
            x1, y1, x2, y2, conf, cls = det
            label = coco_labels[int(cls)]
            if conf > 0.5 and label != "person":
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                cv2.putText(frame, label, (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        hand_results = model_hand(frame)
        if hand_results and hand_results[0].boxes is not None:
            for box in hand_results[0].boxes.xyxy.cpu().numpy():
                x1, y1, x2, y2 = map(int, box[:4])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, "hand", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

def guide_bracelet_to_object(target_cls):
    global bracelet_controller, belt_controller

    if bracelet_controller is None:
        logging.info("Initializing bracelet controller...")
        bracelet_controller = BraceletController(vibration_intensities={
            'bottom': 50, 'top': 50, 'left': 50, 'right': 50
        })

    if not belt_controller:
        connected, belt_controller = connect_belt()
        if not connected:
            raise Exception("Failed to connect to belt.")
        
        # Ensure belt is in APP mode after connection
        belt_controller.set_belt_mode(BeltMode.APP_MODE)
        current_mode = belt_controller.get_belt_mode()
        logging.info(f"Current belt mode: {current_mode}")
        
        # Test vibration with all required parameters for pybelt 1.2.4
        try:
            belt_controller.send_vibration_command(
                channel_index=0,
                pattern=BeltVibrationPattern.CONTINUOUS,
                intensity=70,
                orientation_type=BeltOrientationType.ANGLE,
                orientation=90,
                pattern_iterations=1,
                pattern_period=1000,
                pattern_start_time=0,
                exclusive_channel=False,
                clear_other_channels=False
            )
            time.sleep(1)
            belt_controller.stop_vibration()
            logging.info("Vibration test completed successfully")
        except Exception as e:
            logging.error(f"Vibration test failed: {e}")

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise Exception("Camera open failed.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model_yolo(frame)
        detections = results.xyxy[0].cpu().numpy()
        frame_height, frame_width, _ = frame.shape

        bboxes = []
        for det in detections:
            if len(det) < 6:
                continue
            x1, y1, x2, y2, conf, cls = det
            label = coco_labels[int(cls)]
            if conf > 0.5 and label == target_cls:
                bboxes.append([x1, y1, x2 - x1, y2 - y1])

        if not bboxes:
            continue

        target_bbox = bboxes[0]
        horizontal, vertical = calculate_vibration_direction(target_bbox, frame_width, frame_height)
        
        # Vibrate based on direction
        if horizontal != "center" or vertical != "center":
            angle = {
                "top": 0,
                "right": 90,
                "bottom": 180,
                "left": 270
            }.get(horizontal if horizontal != "center" else vertical, 0)
            
            try:
                belt_controller.send_vibration_command(
                    channel_index=0,
                    pattern=BeltVibrationPattern.CONTINUOUS,
                    intensity=70,
                    orientation_type=BeltOrientationType.ANGLE,
                    orientation=angle,
                    pattern_iterations=1,
                    pattern_period=1000,
                    pattern_start_time=0,
                    exclusive_channel=False,
                    clear_other_channels=False
                )
            except Exception as e:
                logging.error(f"Direction vibration failed: {e}")

        if horizontal == "center" and vertical == "center":
            # Target reached - use pulsating pattern on all motors
            try:
                belt_controller.send_vibration_command(
                    channel_index=0,
                    pattern=BeltVibrationPattern.BEACON,  # Changed from PULSE to BEACON
                    intensity=100,
                    orientation_type=BeltOrientationType.BINARY_MASK,
                    orientation=0b111111,
                    pattern_iterations=3,
                    pattern_period=500,
                    pattern_start_time=0,
                    exclusive_channel=False,
                    clear_other_channels=False
                )
                time.sleep(1.5)  # Let the pattern complete
                belt_controller.stop_vibration()
            except Exception as e:
                logging.error(f"Target reached vibration failed: {e}")
            break

        time.sleep(0.1)  # Faster response time
    
    cap.release()

def calculate_vibration_direction(bbox, frame_width, frame_height):
    x_center = bbox[0] + bbox[2] / 2
    y_center = bbox[1] + bbox[3] / 2

    horizontal = "left" if x_center < frame_width / 3 else "right" if x_center > 2 * frame_width / 3 else "center"
    vertical = "top" if y_center < frame_height / 3 else "bottom" if y_center > 2 * frame_height / 3 else "center"
    return horizontal, vertical

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)