from ultralytics import YOLO
import numpy as np

# Simple tracker (basic ID assignment)
class Tracker:
    def __init__(self):
        self.id_count = 0
        self.objects = {}

    def update(self, detections):
        updated_objects = {}

        for det in detections:
            x1, y1, x2, y2, label = det

            # Assign new ID (basic version)
            self.id_count += 1
            updated_objects[self.id_count] = (x1, y1, x2, y2, label)

        self.objects = updated_objects
        return self.objects


model = YOLO("yolov8n.pt")
tracker = Tracker()

def detect_and_track(frame):
    results = model(frame)
    detections = []

    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls_id = int(box.cls[0])
        label = model.names[cls_id]

        detections.append([x1, y1, x2, y2, label])

    tracked_objects = tracker.update(detections)

    for obj_id, (x1, y1, x2, y2, label) in tracked_objects.items():
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(frame, f"{label} ID:{obj_id}", (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    return frame, tracked_objects