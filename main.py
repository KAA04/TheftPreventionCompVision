"""Simple camera preview with object detection."""

from __future__ import annotations


def main() -> None:
	try:
		import cv2
	except ModuleNotFoundError:
		print("OpenCV is not installed. Run: pip install opencv-python")
		return

	try:
		from ultralytics import YOLO
	except ModuleNotFoundError:
		print("Ultralytics is not installed. Run: pip install ultralytics")
		return

	cap = cv2.VideoCapture(0)
	if not cap.isOpened():
		print("Could not open the default camera.")
		return

	model = YOLO("yolov8n.pt")

	# YOLOv8 COCO labels
	target_labels = {"person", "cell phone", "suitcase", "fork", "knife", "spoon", "tv", "laptop", "mouse", "remote", "keyboard", "microwave", "oven", "toaster", "sink", "refrigerator"}
	label_aliases = {"cell phone": "phone"}
	label_colors = {
		"person": (0, 140, 255),
		"cell phone": (0, 220, 0)#,
		# "suitcase": (180, 105, 255),
		# "fork": (255, 180, 0),
		# "knife": (255, 0, 120),
		# "spoon": (200, 200, 0),
		# "tv": (255, 0, 0),
		# "laptop": (0, 200, 255),
		# "mouse": (120, 200, 120),
		# "remote": (0, 255, 200),
		# "keyboard": (0, 120, 255),
		# "microwave": (255, 120, 0),
		# "oven": (255, 80, 80),
		# "toaster": (160, 160, 255),
		# "sink": (120, 120, 120),
		# "refrigerator": (255, 255, 0),
	}

	def color_from_label(name: str) -> tuple[int, int, int]:
		if name in label_colors:
			return label_colors[name]
		seed = sum(ord(ch) for ch in name) % 255
		return (seed, (seed * 2) % 255, (seed * 3) % 255)

	print("Object detection running. Press 'q' or ESC to quit.")

	while True:
		ok, frame = cap.read()
		if not ok:
			print("Failed to read frame from camera.")
			break

		results = model(frame, verbose=False)
		detections = results[0]

		# Check for collisions and display message
		person_boxes = []
		other_boxes = []

		for box in detections.boxes:
			class_id = int(box.cls[0])
			label = detections.names[class_id]
			if label not in target_labels:
				continue

			confidence = float(box.conf[0])
			if confidence < 0.4:
				continue

			x1, y1, x2, y2 = box.xyxy[0].tolist()
			if label == "person":
				person_boxes.append((x1, y1, x2, y2))
			else:
				other_boxes.append((label, x1, y1, x2, y2))

			label_text = f"{label_aliases.get(label, label)} {confidence:.2f}"
			color = color_from_label(label)

			cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
			cv2.putText(
				frame,
				label_text,
				(int(x1), max(int(y1) - 8, 20)),
				cv2.FONT_HERSHEY_SIMPLEX,
				0.6,
				color,
				2,
			)

		# Check for intersections
		for px1, py1, px2, py2 in person_boxes:
			for label, ox1, oy1, ox2, oy2 in other_boxes:
				if px1 < ox2 and px2 > ox1 and py1 < oy2 and py2 > oy1:  # Intersection condition
					cv2.putText(
						frame,
						f"Person is intersecting with {label}",
						(10, 30),  # Position trxt will appear on the screen
						cv2.FONT_HERSHEY_SIMPLEX,
						0.8,
						(0, 0, 255),
						2,
					)

		cv2.imshow("Object Detection", frame)

		key = cv2.waitKey(1) & 0xFF
		if key in (ord("q"), 27):
			break

	cap.release()
	cv2.destroyAllWindows()


if __name__ == "__main__":
	main()
