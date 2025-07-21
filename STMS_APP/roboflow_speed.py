# dashboard/roboflow_speed.py

import os
import cv2
import numpy as np
from datetime import datetime
from shapely.geometry import LineString
from time import time
import threading
from ultralytics.utils.plotting import Annotator, colors
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from .database import insert_into_database

class RoboflowSpeedEstimator:
    def __init__(self):
        self.line_width = 2
        self.region = [(200, 400), (1000, 400)]  # customize this
        self.spd = {}
        self.trkd_ids = []
        self.trk_pt = {}
        self.trk_pp = {}
        self.saved_ids = set()
        self.gemini_model = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
        os.makedirs("crop", exist_ok=True)

    def process_frame(self, frame, predictions):
        im0 = frame.copy()
        annotator = Annotator(im0, line_width=self.line_width)

        pts = np.array(self.region, np.int32).reshape((-1, 1, 2))
        cv2.polylines(im0, [pts], isClosed=False, color=(104, 0, 123), thickness=3)

        for pred in predictions:
            box = [
                int(pred["x"] - pred["width"] / 2),
                int(pred["y"] - pred["height"] / 2),
                int(pred["x"] + pred["width"] / 2),
                int(pred["y"] + pred["height"] / 2),
            ]
            track_id = pred.get("id", hash(str(box))) % 10000  # fallback if no ID

            self.store_speed(track_id, box)
            speed = self.spd.get(track_id, 0)
            label = f"ID:{track_id} {speed} km/h"
            annotator.box_label(box, label=label, color=colors(track_id, True))

            # Save and analyze once
            if track_id in self.spd and track_id not in self.saved_ids:
                x1, y1, x2, y2 = box
                cropped_image = im0[y1:y2, x1:x2]
                if cropped_image.size > 0:
                    filename = f"crop/{track_id}_{speed}kmh.jpg"
                    cv2.imwrite(filename, cropped_image)

                    threading.Thread(
                        target=self.analyze_and_save_response,
                        args=(filename, track_id, speed, datetime.now()),
                        daemon=True
                    ).start()

                    self.saved_ids.add(track_id)

        return im0

    def store_speed(self, track_id, box):
        if track_id not in self.trk_pt:
            self.trk_pt[track_id] = time()
        if track_id not in self.trk_pp:
            self.trk_pp[track_id] = box

        prev = self.trk_pp[track_id]
        curr = box

        if LineString([prev[:2], curr[:2]]).intersects(LineString(self.region)):
            delta_t = time() - self.trk_pt[track_id]
            if delta_t > 0:
                speed = np.linalg.norm(np.array(curr[:2]) - np.array(prev[:2])) / delta_t
                self.spd[track_id] = round(speed)

        self.trk_pt[track_id] = time()
        self.trk_pp[track_id] = box

    def analyze_and_save_response(self, image_path, track_id, speed, timestamp):
        try:
            with open(image_path, "rb") as img_file:
                base64_image = base64.b64encode(img_file.read()).decode("utf-8")

            message = HumanMessage(
                content=[
                    {"type": "text", "text": "Extract ONLY these details:\n"
                     "| Vehicle Model | Color | Company | Number Plate |\n"
                     "|--------------|--------|---------|--------------|"},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ]
            )

            response = self.gemini_model.invoke([message])
            response_text = response.content.strip()
            valid_rows = [
                row.split("|")[1:-1]
                for row in response_text.split("\n")
                if "|" in row and "Vehicle Model" not in row
            ]
            vehicle_info = valid_rows[0] if valid_rows else ["Unknown"] * 4
            insert_into_database(track_id, speed, timestamp, *vehicle_info)

        except Exception as e:
            print(f"❌ Gemini AI error: {e}")
