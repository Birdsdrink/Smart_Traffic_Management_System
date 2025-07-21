import cv2
import os
from datetime import datetime

# Directory to save cropped vehicle images
SAVE_DIR = "vehicle_captures"
os.makedirs(SAVE_DIR, exist_ok=True)

# Track which vehicles have been captured
captured_vehicle_ids = set()

def capture_vehicle_screenshots(frame, detections):
    """
    Save cropped bounding boxes for vehicles (only once per unique detection_id).
    
    :param frame: Current video frame (BGR)
    :param detections: Roboflow-style Detections object with:
                       - .xyxy
                       - .data['detection_id']
                       - .data['class_name']
                       - .confidence
    """
    xyxy_list = detections.xyxy
    detection_ids = detections.data['detection_id']
    class_names = detections.data['class_name']
    confidences = detections.confidence

    for i, bbox in enumerate(xyxy_list):
        class_name = class_names[i]
        track_id = detection_ids[i]
        confidence = confidences[i]

        # Optional confidence threshold and class filter
        if confidence < 0.6 or class_name != 'car':
            continue

        if track_id in captured_vehicle_ids:
            continue  # Skip if already captured

        # Extract bounding box and crop
        x1, y1, x2, y2 = map(int, bbox)
        crop = frame[y1:y2, x1:x2]

        # Generate unique filename using timestamp and ID
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"{timestamp}_car_{track_id}.jpg"
        filepath = os.path.join(SAVE_DIR, filename)

        # Save cropped image
        cv2.imwrite(filepath, crop)
        print(f"Captured: {filepath}")

        # Mark this vehicle as captured
        captured_vehicle_ids.add(track_id)
