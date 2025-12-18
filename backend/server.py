import cv2
import base64
import numpy as np
import json
import argparse
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from contextlib import asynccontextmanager

from card_detection import CardDetector
from yolo_detection import YOLOCardDetector
from trackers.tracker import IDTracker
from trackers.tracker_deepsort import DeepSORTTracker
from trackers.tracker_bytetrack import ByteTrackTracker

# 1. Global detector
# detector = CardDetector()
detector = YOLOCardDetector(model_path="../yolo_training/mtg_train_gpu_9928171/weights/best.pt", conf_threshold=0.3)

# 2. Use a dictionary or a class to hold the tracker instance globally
# This ensures it's accessible across the async functions
state = {"tracker": None}

async def process_frame(frame_data: str):
    try:
        encoded_data = frame_data.split(',')[1]
        img_data = base64.b64decode(encoded_data)
        nparr = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return []
        
        h, w = frame.shape[:2]
        detections = detector.detect(frame)
        
        tracker = state["tracker"] # Access the initialized tracker
        
        # Dynamic update call based on tracker type
        if isinstance(tracker, DeepSORTTracker):
            tracks_dict = tracker.update(detections, frame)
        else:
            tracks_dict = tracker.update(detections)
        
        formatted_tracks = []
        for track_id, data in tracks_dict.items():
            bx, by, bw, bh = data["bbox"]
            formatted_tracks.append({
                'trackId': track_id,
                'name': f"Card {track_id}",
                'bbox': {
                    'x': float(bx / w), 'y': float(by / h),
                    'w': float(bw / w), 'h': float(bh / h)
                },
                'centroid': {
                    'x': float(data["centroid"][0] / w),
                    'y': float(data["centroid"][1] / h)
                }
            })
        return formatted_tracks
    except Exception as e:
        print(f"Error processing frame: {e}")
        return []

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

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
        print("Client disconnected")

@app.get("/")
async def get_index():
    return FileResponse("../frontend/index.html")

app.mount("/", StaticFiles(directory="../frontend/"), name="static")

def get_tracker(tracker_type):
    if tracker_type == "deepsort":
        print("Using Tracker: DeepSORT")
        return DeepSORTTracker()
    elif tracker_type == "bytetrack":
        print("Using Tracker: ByteTrack")
        return ByteTrackTracker()
    else:
        print("Using Tracker: Default IDTracker")
        return IDTracker(max_disapeared=15, max_distance=80)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("-d", "--deepsort", action="store_const", dest="tracker", const="deepsort")
    group.add_argument("-b", "--bytetrack", action="store_const", dest="tracker", const="bytetrack")
    parser.set_defaults(tracker="default")
    
    args = parser.parse_args()

    # Initialize the tracker BEFORE starting uvicorn
    state["tracker"] = get_tracker(args.tracker)
    
    uvicorn.run(app, host="127.0.0.1", port=args.port)