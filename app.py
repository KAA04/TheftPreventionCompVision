from flask import Flask, Response, jsonify
import cv2
import time
from datetime import datetime
from flask_cors import CORS
import random
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

app = Flask(__name__)
CORS(app)

# Initialize camera
camera = cv2.VideoCapture(0)

# Load YOLO model
model = YOLO("yolov8n.pt")

# Initialize DeepSORT tracker
tracker = DeepSort(max_age=10, n_init=3, nn_budget=10)

# YOLOv8 COCO labels
target_labels = {"person", "cell phone"}
label_aliases = {"cell phone": "phone"}
label_colors = {
    "person": (0, 140, 255),
    "cell phone": (0, 220, 0),
}

def color_from_label(name: str) -> tuple[int, int, int]:
    if name in label_colors:
        return label_colors[name]
    seed = sum(ord(ch) for ch in name) % 255
    return (seed, (seed * 2) % 255, (seed * 3) % 255)

# In-memory event log
object_present = False

event_log = []  # New list to store event timestamps

# Global variable to store theft events
theft_events = []

@app.route('/video_feed')
def video_feed():
    """Stream live video feed with detection and tracking."""
    def generate():
        phone_present = False
        person_interacting = False

        while True:
            success, frame = camera.read()
            if not success:
                break

            results = model(frame, verbose=False)
            detections = results[0]

            # Prepare detections for DeepSORT
            dets = []
            for box in detections.boxes:
                class_id = int(box.cls[0])
                label = detections.names[class_id]
                if label not in target_labels:
                    continue

                confidence = float(box.conf[0])
                if confidence < 0.4:
                    continue

                x1, y1, x2, y2 = box.xyxy[0].tolist()
                dets.append(((x1, y1, x2, y2), confidence, label))

            # Update tracker with detections
            tracks = tracker.update_tracks(dets, frame=frame)

            # Reset state for this frame
            person_boxes = []
            phone_boxes = []

            for track in tracks:
                if not track.is_confirmed():
                    continue

                track_id = track.track_id
                ltrb = track.to_ltrb()
                x1, y1, x2, y2 = map(int, ltrb)
                label = track.get_det_class()

                if label == "person":
                    person_boxes.append((x1, y1, x2, y2))
                elif label == "cell phone":
                    phone_boxes.append((x1, y1, x2, y2))

                label_text = f"{label_aliases.get(label, label)}: {confidence:.2f}"
                color = color_from_label(label)

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    frame,
                    label_text,
                    (x1, max(y1 - 8, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2,
                )

            # Update phone presence state
            phone_present = len(phone_boxes) > 0

            # Check for interactions
            interaction_detected = False
            for px1, py1, px2, py2 in person_boxes:
                for ox1, oy1, ox2, oy2 in phone_boxes:
                    if px1 < ox2 and px2 > ox1 and py1 < oy2 and py2 > oy1:  # Intersection condition
                        interaction_detected = True
                        person_interacting = True
                        object_present = True
                        cv2.putText(
                            frame,
                            "Person interacting with phone",
                            (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (0, 255, 255),
                            2,
                        )

            # Theft detection logic
            if not interaction_detected and person_interacting:
                person_interacting = False
                if not phone_present:
                    theft_event = {
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "message": "ALERT: Phone stolen!"
                    }
                    object_present = False
                    theft_events.append(theft_event)
                    event_log.append(f"{theft_event['timestamp']} - THEFT")  # Add timestamp and type to event_log
                    cv2.putText(
                        frame,
                        theft_event["message"],
                        (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 0, 255),
                        2,
                    )
                else:
                    cv2.putText(
                        frame,
                        "Phone is safe.",
                        (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 0),
                        2,
                    )

            # Encode the frame as JPEG
            _, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            # Stream the frame
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Add endpoint to retrieve event log
@app.route('/event_log', methods=['GET'])
def get_event_log():
    """Retrieve the event log."""
    return jsonify(event_log)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)