from deep_sort_realtime.deepsort_tracker import DeepSort

class DeepSORTTracker:
    def __init__(self, max_age=30, n_init=3):
        """
        max_age: How many frames to keep a 'lost' card before deleting its ID.
        n_init:  How many consecutive frames a card must be seen to get a stable ID.
        """
        self.tracker = DeepSort(
            max_age=30,
            # OPTIMIZATION: Use a smaller, faster embedder
            embedder="mobilenet", 
            # OPTIMIZATION: Half-precision makes it 2x faster on some hardware
            half=True,
            # OPTIMIZATION: Limit the internal "memory" of the AI
            nn_budget=100 
        )

    def update(self, detections, frame):
        """
        detections: List of dicts from CardDetector containing "bbox": (x, y, w, h)
        frame: The raw BGR image (needed for Deep Appearance descriptors)
        """
        # 1. Format detections for DeepSORT: [([x, y, w, h], confidence, class_name), ...]
        raw_detections = []
        for det in detections:
            bbox = list(det["bbox"]) # [x, y, w, h]
            raw_detections.append((bbox, 0.95, "card"))

        # 2. Update the internal Kalman filters and Re-ID tracks
        tracks = self.tracker.update_tracks(raw_detections, frame=frame)
        
        # 3. Convert DeepSORT track objects back into your app's format
        active_tracks = {}
        for track in tracks:
            if not track.is_confirmed():
                continue
            
            track_id = track.track_id
            # to_ltwh() returns [left, top, width, height]
            ltwh = track.to_ltwh()
            
            active_tracks[track_id] = {
                "bbox": (int(ltwh[0]), int(ltwh[1]), int(ltwh[2]), int(ltwh[3])),
                "centroid": (ltwh[0] + ltwh[2]/2, ltwh[1] + ltwh[3]/2)
            }
            
        return active_tracks