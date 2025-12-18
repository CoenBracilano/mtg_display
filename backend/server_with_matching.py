import cv2
import base64
import numpy as np
import json
import argparse
import uvicorn
import asyncio
import os
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from card_detection import CardDetector
from yolo_detection import YOLOCardDetector
from trackers.tracker import IDTracker
from trackers.tracker_deepsort import DeepSORTTracker
from trackers.tracker_bytetrack import ByteTrackTracker
from card_matching import CardMatcher

# --- CONFIGURATION & STATE ---
detector = YOLOCardDetector(
    model_path="../yolo_training/model_weights/9928394_best.pt", 
    conf_threshold=0.3
)
matcher = CardMatcher(db_path="../card_database")
executor = ThreadPoolExecutor(max_workers=4) # Increased for retries

state = {
    "tracker": None,
    "matched_cards": {},
    "previous_tracks": set(),
    "pending_matches": set(),
    "match_attempts": {},  # track_id -> int
    "max_retries": 3
}

# --- ENHANCEMENT PIPELINE ---

def enhance_for_matching(image):
    """Applies CLAHE contrast and Sharpening to assist SIFT/ORB feature detection"""
    if image is None or image.size == 0:
        return image
        
    # 1. Contrast Enhancement (CLAHE)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    enhanced = cv2.merge((cl, a, b))
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
    
    # 2. Sharpening
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    enhanced = cv2.filter2D(enhanced, -1, kernel)
    
    return enhanced

def normalize_card_crop(card_crop, target_width=480, target_height=680):
    """Standardize size for the matcher"""
    if card_crop is None or card_crop.size == 0:
        return None
    return cv2.resize(card_crop, (target_width, target_height), interpolation=cv2.INTER_CUBIC)

# --- CORE LOGIC ---

def match_card_sync(track_id, card_crop, attempt=1):
    """Synchronous matching function used by the executor"""
    try:
        # Step 1: Enhance if it's a retry
        processed = card_crop
        if attempt > 1:
            processed = enhance_for_matching(card_crop)
            
        # Step 2: Normalize size
        normalized = normalize_card_crop(processed)
        if normalized is None:
            return track_id, None
        
        # Step 3: Identify via SIFT/ORB
        # We lower the requirements slightly for attempt 1, increase them for retries
        min_matches = 12 if attempt == 1 else 15
        result = matcher.identify(normalized, min_matches=min_matches, dist_threshold=50)
        return track_id, result
    except Exception as e:
        print(f"Error matching track {track_id} on attempt {attempt}: {e}")
        return track_id, None

def match_and_update(track_id, card_crop, loop):
    """Recursive retry logic - now thread-safe"""
    current_attempt = state["match_attempts"].get(track_id, 1)
    
    # Try to match (Synchronous work)
    _, result = match_card_sync(track_id, card_crop, attempt=current_attempt)
    
    if result:
        state["matched_cards"][track_id] = result
        state["pending_matches"].discard(track_id)
        state["match_attempts"].pop(track_id, None)
        print(f"✨ Match Found (ID {track_id}): {result['name']}")
    else:
        if current_attempt < state["max_retries"]:
            state["match_attempts"][track_id] = current_attempt + 1
            print(f"🔄 ID {track_id} retry {state['match_attempts'][track_id]}/{state['max_retries']}")
            
            # 2. Use the passed loop to schedule the next attempt safely
            loop.call_soon_threadsafe(
                lambda: loop.run_in_executor(executor, match_and_update, track_id, card_crop, loop)
            )
        else:
            state["matched_cards"][track_id] = {'name': 'Unknown Card', 'confidence': 0}
            state["pending_matches"].discard(track_id)
            state["match_attempts"].pop(track_id, None)

# --- FRAME PROCESSING ---

async def process_frame(frame_data: str):
    try:
        # Decode base64
        encoded_data = frame_data.split(',')[1]
        img_data = base64.b64decode(encoded_data)
        nparr = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None: return []
        
        h, w = frame.shape[:2]
        detections = detector.detect(frame)
        
        # Update tracker
        tracker = state["tracker"]
        tracks_dict = tracker.update(detections, frame) if isinstance(tracker, DeepSORTTracker) else tracker.update(detections)
        
        current_track_ids = set(tracks_dict.keys())
        
        # Capture New Tracks
        new_track_ids = current_track_ids - state["previous_tracks"] - state["pending_matches"]
        
        for tid in new_track_ids:
            if tid in tracks_dict:
                bx, by, bw, bh = tracks_dict[tid]["bbox"]
                # Safeguard crop area
                x1, y1 = max(0, int(bx)), max(0, int(by))
                x2, y2 = min(w, int(bx + bw)), min(h, int(by + bh))
                
                if x2 > x1 and y2 > y1:
                    state["pending_matches"].add(tid)
                    state["match_attempts"][tid] = 1
                    card_crop = frame[y1:y2, x1:x2].copy()
                    
                    # Get the loop here (main thread)
                    current_loop = asyncio.get_event_loop()
                    
                    # Pass 'current_loop' as the third argument
                    current_loop.run_in_executor(executor, match_and_update, tid, card_crop, current_loop)

        state["previous_tracks"] = current_track_ids
        
        # Cleanup
        disappeared = set(state["matched_cards"].keys()) - current_track_ids
        for tid in disappeared:
            state["pending_matches"].discard(tid)
            state["matched_cards"].pop(tid, None)
            state["match_attempts"].pop(tid, None)

        # Build Response
        formatted = []
        for tid, data in tracks_dict.items():
            bx, by, bw, bh = data["bbox"]
            info = state["matched_cards"].get(tid)
            
            # Status messages
            if tid in state["pending_matches"]:
                display_name = f"Matching (Try {state['match_attempts'].get(tid, 1)}...)"
            else:
                display_name = info['name'] if info else "Detecting..."

            formatted.append({
                'trackId': tid,
                'name': display_name,
                'bbox': {'x': bx/w, 'y': by/h, 'w': bw/w, 'h': bh/h},
                'confidence': info.get('confidence', 0) if info else 0,
                'url': info.get('url', '') if info else ''
            })
        return formatted
        
    except Exception as e:
        print(f"Frame processing error: {e}")
        return []

# --- FASTAPI SERVER ---
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            if message['type'] == 'frame':
                tracks = await process_frame(message['data'])
                await websocket.send_json({'type': 'tracks', 'tracks': tracks})
    except WebSocketDisconnect:
        pass

@app.get("/")
async def get_index():
    return FileResponse("../frontend/index.html")

app.mount("/", StaticFiles(directory="../frontend/"), name="static")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--deepsort", action="store_const", dest="tracker", const="deepsort")
    parser.add_argument("-b", "--bytetrack", action="store_const", dest="tracker", const="bytetrack")
    parser.set_defaults(tracker="default")
    args = parser.parse_args()

    if args.tracker == "deepsort": state["tracker"] = DeepSORTTracker()
    elif args.tracker == "bytetrack": state["tracker"] = ByteTrackTracker()
    else: state["tracker"] = IDTracker(max_disapeared=15, max_distance=80)

    uvicorn.run(app, host="127.0.0.1", port=8000)