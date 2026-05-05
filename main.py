from flask import Flask, Response, jsonify
import cv2
import time
from datetime import datetime
from flask_cors import CORS
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

app = Flask(__name__)
CORS(app)

camera = cv2.VideoCapture(0)

model = YOLO("yolov8n.engine")

tracker = DeepSort(max_age=10, n_init=3, nn_budget=10)

target_labels = {"person", "cell phone"}
label_aliases = {"cell phone": "phone"}

label_colors = {
    "person": (0, 140, 255),
    "cell phone": (0, 220, 0),
}

def color_from_label(name: str):
    seed = sum(ord(ch) for ch in name) % 255
    return (seed, (seed * 2) % 255, (seed * 3) % 255)

event_log = []
theft_events = []

# ===========================
# STATE VARIABLES
# ===========================
interaction_started = False
was_overlapping = False

phone_missing_start = None
THEFT_DELAY = 5  # seconds


@app.route('/video_feed')
def video_feed():

    def generate():
        global interaction_started, was_overlapping, phone_missing_start

        while True:
            success, frame = camera.read()
            if not success:
                break

            frame = cv2.resize(frame, (640, 640))

            # ===========================
            # YOLO inference
            # ===========================
            results = model(frame, imgsz=640, verbose=False)
            detections = results[0]

            dets = []

            for box in detections.boxes:
                class_id = int(box.cls[0])
                label = detections.names[class_id]

                if label not in target_labels:
                    continue

                conf = float(box.conf[0])
                if conf < 0.4:
                    continue

                x1, y1, x2, y2 = box.xyxy[0].tolist()
                dets.append(((x1, y1, x2, y2), conf, label))
                
            print(dets)

            # ===========================
            # Parse detections
            # ===========================
            person_boxes = []
            phone_boxes = []

            for (box, conf, label) in dets:
                x1, y1, x2, y2 = map(int, box)

                color = color_from_label(label)

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                text = f"{label_aliases.get(label, label)} {conf:.2f}"

                cv2.putText(
                    frame,
                    text,
                    (x1, max(y1 - 8, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2,
                )

                if label == "person":
                    person_boxes.append((x1, y1, x2, y2))
                elif label == "cell phone":
                    phone_boxes.append((x1, y1, x2, y2))

            person_present = len(person_boxes) > 0
            phone_present = len(phone_boxes) > 0

            # ===========================
            # STEP 1: detect overlap
            # ===========================
            overlap_detected = False

            for px1, py1, px2, py2 in person_boxes:
                for ox1, oy1, ox2, oy2 in phone_boxes:
                    if px1 < ox2 and px2 > ox1 and py1 < oy2 and py2 > oy1:
                        overlap_detected = True
                        interaction_started = True
                        was_overlapping = True

                        message = "Person interacting with phone"
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        font_scale = 0.8
                        thickness = 2
                        
                        # (text_width, text_height), baseline = cv2.getTextSize(
                        #     message, font, font_scale, thickness
                        # )
                        
                        frame_height, frame_width = frame.shape[:2]
                        x = (frame_width // 4)
                        y = frame_height - (frame_height // 6)  # slightly above bottom so it doesn't overlap UI

                        cv2.putText(
                            frame,
                            message,
                            (x, y),
                            font,
                            font_scale,
                            (0, 255, 255),
                            thickness,
                        )

            # ===========================
            # STEP 2: PHONE DISAPPEAR TIMER
            # ===========================
            if interaction_started and was_overlapping:

                if phone_present:
                    # reset timer if phone reappears
                    phone_missing_start = None
                    elapsed = 0

                    if not overlap_detected:
                        message = "Phone Safe"
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        font_scale = 0.8
                        thickness = 2
                    
                        frame_height, frame_width = frame.shape[:2]
                        x = (frame_width // 4)
                        y = frame_height - (frame_height // 6)  # slightly above bottom so it doesn't overlap UI

                        cv2.putText(
                            frame,
                            message,
                            (x, y),
                            font,
                            font_scale,
                            (255, 255, 0),
                            thickness,
                        )

                else:
                    # start timer when phone disappears
                    if phone_missing_start is None:
                        phone_missing_start = time.time()
                        #print("phone now missing")

                    elapsed = time.time() - phone_missing_start

                    #print("phone missing for:", elapsed)

                    # only trigger theft after 5 seconds
                    if elapsed >= THEFT_DELAY:
                        #print("phone has been stolen!!!!!!!!!!!!!!!!!!!!!!!!")
                        event = {
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "message": "ALERT: Phone stolen!"
                        }

                        theft_events.append(event)
                        event_log.append(f"{event['timestamp']} - THEFT")

                        message = "PHONE STOLEN"
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        font_scale = 0.8
                        thickness = 2
                    
                        frame_height, frame_width = frame.shape[:2]
                        x = (frame_width // 4)
                        y = frame_height - (frame_height // 6)  # slightly above bottom so it doesn't overlap UI
    
                        cv2.putText(
                                frame,
                            message,
                            (x, y),
                            font,
                            font_scale,
                            (0, 0, 255),
                            thickness,
                        )

                        # reset system after theft
                        interaction_started = False
                        was_overlapping = False
                        phone_missing_start = None

            # Encode frame
            _, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/event_log', methods=['GET'])
def get_event_log():
    return jsonify(event_log)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)