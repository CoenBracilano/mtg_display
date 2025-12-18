from ultralytics import YOLO
import numpy as np
import cv2

class YOLOCardDetector:
    def __init__(self, model_path="best.pt", conf_threshold=0.5):
        # Load your custom trained YOLO model
        self.model = YOLO(model_path)
        self.conf = conf_threshold

    def detect(self, frame):
        # Run YOLO inference
        
        results = self.model(frame, stream=True, conf=self.conf, verbose=False)

        frame = self.sharpen_image(frame)
        
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Get coordinates in [x1, y1, x2, y2] (top-left, bottom-right)
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                
                # Convert to (x, y, w, h) for your trackers
                w = x2 - x1
                h = y2 - y1
                
                # Calculate center for IDTracker/ByteTrack
                cx = x1 + w / 2.0
                cy = y1 + h / 2.0
                
                detections.append({
                    "bbox": (int(x1), int(y1), int(w), int(h)),
                    "centroid": (cx, cy),
                    "conf": float(box.conf[0]),
                    "class_id": int(box.cls[0])
                })
        
        return detections
    
    def sharpen_image(self, frame):
        gaussian = cv2.GaussianBlur(frame, (0, 0), 2.0)
        sharpened = cv2.addWeighted(frame, 1.5, gaussian, -0.5, 0)
        return sharpened