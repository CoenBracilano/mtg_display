import numpy as np
import supervision as sv

class ByteTrackTracker:
    def __init__(self):
        # Changed 'track_buffer' to 'lost_track_buffer' for compatibility with newer supervision versions
        self.tracker = sv.ByteTrack(
            track_activation_threshold=0.25, 
            lost_track_buffer=60
        )

    def update(self, detections):
        if not detections:
            return {}

        # Convert detections to xyxy format
        xyxy = []
        confidences = []
        for det in detections:
            x, y, w, h = det["bbox"]
            xyxy.append([x, y, x + w, y + h])
            confidences.append(0.9) # High confidence for contour detections
            
        sv_detections = sv.Detections(
            xyxy=np.array(xyxy, dtype=np.float32),
            confidence=np.array(confidences, dtype=np.float32),
            class_id=np.zeros(len(detections), dtype=int)
        )

        # Update Tracker
        tracked_detections = self.tracker.update_with_detections(sv_detections)
        
        # Format for frontend
        active_tracks = {}
        # Ensure we have tracker_id available
        if tracked_detections.tracker_id is not None:
            for i in range(len(tracked_detections)):
                box = tracked_detections.xyxy[i]
                tid = tracked_detections.tracker_id[i]
                
                w = box[2] - box[0]
                h = box[3] - box[1]
                active_tracks[int(tid)] = {
                    "bbox": (int(box[0]), int(box[1]), int(w), int(h)),
                    "centroid": (float(box[0] + w/2), float(box[1] + h/2))
                }
            
        return active_tracks