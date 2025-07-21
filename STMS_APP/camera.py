import cv2

class VideoCamera:
    def __init__(self):
        self.cap = cv2.VideoCapture("static/videos/tc.mp4")

    def get_frame(self):
        success, frame = self.cap.read()
        if not success:
            return None
        return frame

    def release(self):
        if self.cap.isOpened():
            self.cap.release()
