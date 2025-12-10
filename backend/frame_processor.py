import cv2
import numpy as np
from card_detection import CardDetector
from tracker import IDTracker
from card_matching import CardMatcher


class FrameProcessor:    
    def __init__(self, detector_params=None, tracker_params=None):

        # Initialize matcher
        # Hardcoding database path for now (can change it later)
        self.matcher = CardMatcher("../card_database")
        
        # Initialize detector
        if detector_params is None:
            detector_params = {'min_area': 2000, 'max_area': 500000, 'epsilon_factor': 0.05}
        self.detector = CardDetector(**detector_params)
        
        # Initialize tracker
        if tracker_params is None:
            tracker_params = {'max_disapeared': 30, 'max_distance': 80}
        self.tracker = IDTracker(**tracker_params)
        
        # Cache: {track_id: {'name': str, 'confidence': float, ...}}
        self.identified_tracks = {}
        
        # Matching parameters
        self.min_matches = 15
        self.dist_threshold = 50
    
    def process_frame(self, frame):
        """
        Process a single frame
        
        Args:
            frame: OpenCV BGR image
            
        Returns:
            Annotated frame with bounding boxes and card names
        """
        # Step 1: Detect cards
        detections = self.detector.detect(frame)
        
        # Step 2: Update tracker
        tracks = self.tracker.update(detections)
        
        # Step 3: Process each track
        output = frame.copy()
        
        for track_id, track_data in tracks.items():
            # Find corresponding detection
            detection = self._find_detection_for_track(track_data, detections)
            
            # Check cache first
            if track_id in self.identified_tracks:
                # Already identified - use cached result
                card_info = self.identified_tracks[track_id]
                self._draw_card(output, track_data, card_info, detection)
            
            elif detection is not None:
                # New track with detection - try to identify
                warped = self._warp_card(frame, detection)
                result = self.matcher.identify(warped, self.min_matches, self.dist_threshold)
                
                if result:
                    # Cache the result
                    self.identified_tracks[track_id] = result
                    self._draw_card(output, track_data, result, detection)
                else:
                    # Couldn't identify yet
                    self._draw_card(output, track_data, None, detection)
            
            else:
                # Track exists but no detection (temporarily obscured)
                # Draw with cached info if available
                cached = self.identified_tracks.get(track_id)
                self._draw_card(output, track_data, cached, None)
        
        # Step 4: Clean up cache for dead tracks
        self._cleanup_cache(tracks)
        
        return output
    
    def _find_detection_for_track(self, track_data, detections, threshold=50):
        """Find detection that matches this track based on centroid"""
        track_centroid = track_data["centroid"]
        
        best_detection = None
        best_distance = threshold
        
        for detection in detections:
            det_centroid = detection["centroid"]
            dist = np.sqrt((track_centroid[0] - det_centroid[0])**2 + 
                          (track_centroid[1] - det_centroid[1])**2)
            
            if dist < best_distance:
                best_distance = dist
                best_detection = detection
        
        return best_detection
    
    def _warp_card(self, frame, detection, width=300, height=420):
        """Warp detected card to rectangular view for better matching"""
        pts = detection["contour"].reshape(4, 2)
        pts = self._order_points(pts)
        
        dst = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], 
                      dtype="float32")
        M = cv2.getPerspectiveTransform(pts.astype("float32"), dst)
        warped = cv2.warpPerspective(frame, M, (width, height))
        
        return warped
    
    def _order_points(self, pts):
        """Order points for perspective transform: top-left, top-right, bottom-right, bottom-left"""
        rect = np.zeros((4, 2), dtype="float32")
        
        s = np.sum(pts, axis=1)
        rect[0] = pts[np.argmin(s)]  # top-left
        rect[2] = pts[np.argmax(s)]  # bottom-right
        
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]  # top-right
        rect[3] = pts[np.argmax(diff)]  # bottom-left
        
        return rect
    
    def _draw_card(self, frame, track_data, card_info, detection):
        """Draw bounding box, card name, and metadata"""
        x, y, w, h = track_data["bbox"]
        cx, cy = track_data["centroid"]
        
        # Draw contour if available
        if detection and "contour" in detection:
            cv2.drawContours(frame, [detection["contour"]], -1, (0, 255, 0), 2)
        
        # Draw bounding box
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Draw centroid
        cv2.circle(frame, (int(cx), int(cy)), 3, (0, 0, 255), -1)
        
        # Prepare label
        if card_info:
            label = card_info['name']
            confidence = card_info.get('confidence', 100)
            
            # Color based on confidence
            if confidence > 80:
                color = (0, 255, 0)  # Green - high confidence
            elif confidence > 60:
                color = (0, 165, 255)  # Orange - medium confidence
            else:
                color = (0, 100, 255)  # Red-orange - low confidence
            
            # Add confidence if not perfect
            if confidence < 95:
                label += f" ({confidence:.0f}%)"
        else:
            label = "Identifying..."
            color = (255, 255, 0)  # Yellow
        
        # Draw label with background
        self._draw_label(frame, label, x, y, color)
    
    def _draw_label(self, frame, text, x, y, color):
        """Draw text label with semi-transparent background"""
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        
        # Get text size
        (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        
        # Position above bbox
        label_y = max(y - 10, text_h + 10)
        
        # Draw background
        overlay = frame.copy()
        cv2.rectangle(overlay,
                     (x, label_y - text_h - 5),
                     (x + text_w + 10, label_y + baseline),
                     (0, 0, 0),
                     -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        # Draw text
        cv2.putText(frame, text, (x + 5, label_y),
                   font, font_scale, color, thickness)
    
    def _cleanup_cache(self, tracks):
        """Remove cached identifications for tracks that no longer exist"""
        active_ids = set(tracks.keys())
        cached_ids = set(self.identified_tracks.keys())
        
        for track_id in cached_ids - active_ids:
            del self.identified_tracks[track_id]
    
    def get_stats(self):
        """Get current processing statistics"""
        return {
            'active_tracks': len(self.tracker.tracks),
            'identified_cards': len(self.identified_tracks),
            'unique_cards': len(set(info['name'] for info in self.identified_tracks.values())),
            'database_size': len(self.matcher.cards)
        }