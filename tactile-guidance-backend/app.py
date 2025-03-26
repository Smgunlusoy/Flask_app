import sys
import os
from flask import Flask, request, jsonify, Response
import cv2
import json
import numpy as np
import torch

# Add the parent directory of aibox to PYTHONPATH
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
aibox_path = os.path.join(project_root, 'aibox')

print(f"Adding {aibox_path} to PYTHONPATH")
sys.path.insert(0, project_root)

# Verify the path is added
print(f"Current PYTHONPATH: {sys.path}")

# Print project_root and aibox_path
print(f"Project root: {project_root}")
print(f"aibox path: {aibox_path}")

try:
    from aibox.controller import TaskController
    from aibox.bracelet import BraceletController
    print("Successfully imported aibox modules")
except ImportError as e:
    print(f"Error importing aibox modules: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

app = Flask(__name__)

# Load the YOLO model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

@app.route('/')
def home():
    return "Welcome to the Tactile Guidance System!"

def generate_frames():
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        raise RuntimeError("Could not open camera.")

    while True:
        success, frame = camera.read()
        if not success:
            break
        
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Perform object detection
        results = model(frame_rgb)

        # Process detection results
        detections = results.xyxy[0].numpy()

        for det in detections:
            x1, y1, x2, y2, conf, cls = det
            # Draw bounding box on the frame
            label = f'Class: {int(cls)}, Score: {conf:.2f}'
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)  # Draw bounding box
            cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)  # Show label

        # Encode the frame in JPEG format
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        # Yield the frame in the proper format for streaming
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/detect_objects', methods=['POST'])
def detect_objects():
    try:
        # Open the webcam
        webcam = cv2.VideoCapture(0)
        if not webcam.isOpened():
            return jsonify({"error": "Error opening OpenCV VideoCapture"}), 500

        ret, frame = webcam.read()
        if not ret:
            return jsonify({"error": "Error reading frame from camera"}), 500

        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Perform object detection
        results = model(frame_rgb)

        # Process detection results
        detections = results.xyxy[0].numpy()

        detected_objects = []
        for det in detections:
            x1, y1, x2, y2, conf, cls = det
            detected_objects.append({
                "class": int(cls),
                "score": float(conf),
                "box": [float(x1), float(y1), float(x2), float(y2)]
            })

        webcam.release()
        return jsonify({"message": "Object detection completed", "objects": detected_objects}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/activate_bracelet', methods=['POST'])
def activate_bracelet():
    participant = request.json.get('participant')
    if not participant:
        return jsonify({"error": "Participant ID not provided"}), 400

    try:
        calibration_file = f'results/calibration/calibration_participant_{participant}.json'
        with open(calibration_file) as file:
            participant_vibration_intensities = json.load(file)
    except FileNotFoundError:
        return jsonify({"error": f"Calibration file {calibration_file} not found"}), 404
    except json.JSONDecodeError:
        return jsonify({"error": "Error decoding calibration file"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    try:
        bracelet_controller = BraceletController(vibration_intensities=participant_vibration_intensities)
        task_controller = TaskController(
            weights_obj='yolov5s.pt',
            weights_hand='hand.pt',
            weights_tracker='osnet_x0_25_market1501.pt',
            weights_depth_estimator='midas_v21_384',
            source='0'
        )

        return jsonify({
            "message": "Bracelet guidance system initialized",
            "participant": participant
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True, use_reloader=False)