import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor

from yolo_detection import YOLOCardDetector
from trackers.tracker import IDTracker
from trackers.tracker_deepsort import DeepSORTTracker
from trackers.tracker_bytetrack import ByteTrackTracker
from card_matching import CardMatcher

# --- CONFIGURATION ---
VIDEO_SOURCE = "../test_files/mtgvideo2.mp4"
OUTPUT_FILE = "../test_files/mtgvideo2_processed.mp4"
WINDOW_NAME = "MTG Tracker - Saving Mode"

detector = YOLOCardDetector(model_path="../yolo_training/model_weights/9928394_best.pt", conf_threshold=0.3)
matcher = CardMatcher(db_path="../card_database")
executor = ThreadPoolExecutor(max_workers=4)

state = {
    "tracker": IDTracker(max_disapeared=15, max_distance=80),
    "matched_cards": {}, 
    "pending_matches": set(),
    "match_attempts": {},
    "max_retries": 3
}

def enhance_image(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    enhanced = cv2.merge((cl, a, b))
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    return cv2.filter2D(enhanced, -1, kernel)

def match_worker(track_id, card_crop):
    attempt = state["match_attempts"].get(track_id, 1)
    processed = card_crop if attempt == 1 else enhance_image(card_crop)
    normalized = cv2.resize(processed, (480, 680))
    result = matcher.identify(normalized, min_matches=15, dist_threshold=50)
    
    if result:
        state["matched_cards"][track_id] = result
        state["pending_matches"].discard(track_id)
        print(f"✨ ID {track_id} matched: {result['name']}")
    elif attempt < state["max_retries"]:
        state["match_attempts"][track_id] = attempt + 1
        executor.submit(match_worker, track_id, card_crop)
    else:
        state["matched_cards"][track_id] = {"name": "Unknown", "confidence": 0}
        state["pending_matches"].discard(track_id)

def run_and_save():
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    
    # 1. Get video properties for the Writer
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)
    
    # 2. Initialize VideoWriter (Using 'mp4v' codec for MP4)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_FILE, fourcc, fps, (width, height))
    
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    previous_tracks = set()

    print(f"Processing... Output will be saved to {OUTPUT_FILE}")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        display_frame = frame.copy()
        
        # Detection & Tracking
        detections = detector.detect(frame)
        tracks_dict = state["tracker"].update(detections)
        current_track_ids = set(tracks_dict.keys())
        
        # Trigger Matching
        new_tracks = current_track_ids - previous_tracks - state["pending_matches"]
        for tid in new_tracks:
            bx, by, bw, bh = tracks_dict[tid]["bbox"]
            x1, y1, x2, y2 = max(0, int(bx)), max(0, int(by)), min(width, int(bx+bw)), min(height, int(by+bh))
            if x2 > x1 and y2 > y1:
                state["pending_matches"].add(tid)
                state["match_attempts"][tid] = 1
                crop = frame[y1:y2, x1:x2].copy()
                executor.submit(match_worker, tid, crop)

        previous_tracks = current_track_ids

        # Rendering
        for tid, data in tracks_dict.items():
            bx, by, bw, bh = map(int, data["bbox"])
            info = state["matched_cards"].get(tid)
            
            if tid in state["pending_matches"]:
                label, color = f"ID {tid}: Matching...", (0, 255, 255)
            elif info:
                label, color = f"{info['name']} ({info['confidence']:.0f}%)", (0, 255, 0)
            else:
                label, color = f"ID {tid}: Detecting...", (0, 165, 255)

            cv2.rectangle(display_frame, (bx, by), (bx + bw, by + bh), color, 2)
            cv2.putText(display_frame, label, (bx, by - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # 3. Write the frame to the output file
        out.write(display_frame)
        
        # Show on screen
        cv2.imshow(WINDOW_NAME, display_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 4. Cleanup
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("Video saving complete.")

if __name__ == "__main__":
    run_and_save()