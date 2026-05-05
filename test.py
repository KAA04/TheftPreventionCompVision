import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# ---------------------------
# Load model (TensorRT engine)
# ---------------------------
model = YOLO("yolov8n.engine")

# ---------------------------
# Tracker (optional but kept since you used it)
# ---------------------------
tracker = DeepSort(max_age=10, n_init=3, nn_budget=10)

# ---------------------------
# Camera
# ---------------------------
cap = cv2.VideoCapture(0)

IMG_SIZE = 640

target_labels = {"person", "cell phone"}
label_aliases = {"cell phone": "phone"}

def color(name):
    seed = sum(ord(c) for c in name) % 255
    return (seed, (seed * 2) % 255, (seed * 3) % 255)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize for YOLO TensorRT engine
    frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))

    # ---------------------------
    # YOLO inference
    # ---------------------------
    results = model(frame, imgsz=IMG_SIZE, verbose=False)
    detections = results[0]

    dets = []

    for box in detections.boxes:
        cls_id = int(box.cls[0])
        label = detections.names[cls_id]

        if label not in target_labels:
            continue

        conf = float(box.conf[0])
        if conf < 0.4:
            continue

        x1, y1, x2, y2 = box.xyxy[0].tolist()
        dets.append(((x1, y1, x2, y2), conf, label))

    # ---------------------------
    # Tracking
    # ---------------------------
    tracks = tracker.update_tracks(dets, frame=frame)

    for t in tracks:
        if not t.is_confirmed():
            continue

        x1, y1, x2, y2 = map(int, t.to_ltrb())
        label = t.get_det_class()

        cv2.rectangle(frame, (x1, y1), (x2, y2), color(label), 2)

        text = f"{label_aliases.get(label, label)} ID:{t.track_id}"
        cv2.putText(frame, text, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color(label), 2)

    # ---------------------------
    # Show preview
    # ---------------------------
    cv2.imshow("Live Camera (YOLO + TensorRT)", frame)

    # press Q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()