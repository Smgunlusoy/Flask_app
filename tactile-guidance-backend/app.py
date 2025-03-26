import sys
import os
from flask import Flask, request, jsonify, Response
import cv2
import json

# Add the parent directory of aibox to PYTHONPATH
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
aibox_path = os.path.join(project_root, 'aibox')

print(f"Adding {aibox_path} to PYTHONPATH")
sys.path.insert(0, project_root)  # Add project root to path

# Verify the path is added
print(f"Current PYTHONPATH: {sys.path}")

# Print project_root and aibox_path
print(f"Project root: {project_root}")
print(f"aibox path: {aibox_path}")

# List contents of project_root and aibox_path
print(f"Contents of project root ({project_root}): {os.listdir(project_root)}")
print(f"Contents of aibox path ({aibox_path}): {os.listdir(aibox_path)}")

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

@app.route('/')
def home():
    return "Welcome to the Tactile Guidance System!"

@app.route('/activate_camera', methods=['POST'])
def activate_camera():
    try:
        webcam = cv2.VideoCapture(0)
        if not webcam.isOpened():
            return jsonify({"error": "Error opening OpenCV VideoCapture"}), 500

        ret, frame = webcam.read()
        if not ret:
            return jsonify({"error": "Error reading frame from camera"}), 500

        webcam.release()
        return jsonify({"message": "Camera activated and frame captured"}), 200
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

def generate_frames():
    camera = cv2.VideoCapture(0)
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True, use_reloader=False)