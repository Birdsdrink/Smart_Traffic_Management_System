from django.shortcuts import render
from inference import InferencePipeline
from .utils import SpeedEstimator
from .screenshot import *
import cv2
import threading
import json


# Global frame buffer
latest_frame = None
pipeline = None
#estimator = SpeedEstimator()


def my_sink(result, video_frame):
    global latest_frame
    if result.get("label_visualization"):
        frame = result["label_visualization"].numpy_image
        detections = result["detections"]
        #frame = estimator.estimate_speed_from_detections(frame, detections)
        



        #predictions = json.loads(predictions)
        capture_vehicle_screenshots(frame, detections)
        #frame = estimator.process_frame(frame, predictions)
        latest_frame = cv2.imencode('.jpg', frame)[1].tobytes()
        print(result) 


def start_pipeline():
    global pipeline
    pipeline = InferencePipeline.init_with_workflow(
        api_key="bWsid6TChQwIXKdKUSXy",
        workspace_name="datalink-ai",
        workflow_id="vehicle-detection",
        video_reference="static/videos/tc.mp4",
        max_fps=30,
        on_prediction=my_sink
    )
    pipeline.start()

# Run pipeline once when server starts
threading.Thread(target=start_pipeline, daemon=True).start()


def generate_frames():
    global latest_frame
    while True:
        if latest_frame is not None:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + latest_frame + b'\r\n')


def generate_frames():
    global latest_frame
    while True:
        if latest_frame is not None:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + latest_frame + b'\r\n')
