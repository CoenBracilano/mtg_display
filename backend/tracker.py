import math
import numpy as np
import cv2

class IDTracker:
    def __init__(self, max_disapeared =15, max_distance= 80):
        self.next_id = 0
        self.tracks = {}
        self.max_disapeared = max_disapeared
        self.max_distance = max_distance

    def _register(self, centroid, bbox):
        self.tracks[self.next_id] = {
                "centroid": centroid,
                "bbox": bbox,
                "disapeared": 0
            }
        self.next_id +=1
    
    def _deregister(self, track_id):
        del self.tracks[track_id]
    
    def update(self, detections):
        
        # if we have no detections then we should increment the disapeared objects counter
        if len(detections) == 0:
            to_delete = []
            for id, track in self.tracks.items():
                track["disapeared"] += 1
                if track["disapeared"] > self.max_disapeared:
                    to_delete.append(id)
            for id in to_delete:
                self._deregister(id)
            return self.tracks
        
        input_centroids = [det["centroid"] for det in detections]
        input_bboxes = [det["bbox"] for det in detections]

        # if we arent tracking anythign then register all our current detected objects
        if len(self.tracks) == 0:
            for c, b in zip(input_centroids, input_bboxes):
                self._register(c,b)
            return self.tracks
        
        # match our detections to our exsisting tracked objects
        track_ids = list(self.tracks.keys())
        track_centroids = [self.tracks[id]["centroid"] for id in track_ids]

        # compute the distance matrix with size [tracks x detections]
        distances = []
        for tc in track_centroids:
            row  = []
            for dc in input_centroids:
                d = math.dist(tc,dc)
                row.append(d)
            distances.append(row)
        
        distances = np.array(distances)
        used_tracks = set()
        used_dets = set()

        while True:
            if distances.size ==0:
                break

            min_idx = np.unravel_index(np.argmin(distances), distances.shape)
            t_idx, d_idx = int(min_idx[0]), int(min_idx[1])
            min_dist = distances[t_idx, d_idx]

            if min_dist > self.max_distance:
                break

            track_id = track_ids[t_idx]
            if track_id in used_tracks or d_idx in used_dets:
                distances[t_idx, d_idx] = float("inf")
                continue

            self.tracks[track_id]["centroid"] = input_centroids[d_idx]
            self.tracks[track_id]["bbox"] = input_bboxes[d_idx]
            self.tracks[track_id]["disapeared"] = 0

            used_tracks.add(track_id)
            used_dets.add(d_idx)

            distances[t_idx, :] = float("inf")
            distances[:, d_idx] = float("inf")
            
        # any tracks that werent matched are disapeared
        # and deregistered if they have been gone too long
        for track_id in track_ids:
            if track_id not in used_tracks:
                self.tracks[track_id]["disapeared"] += 1
                if self.tracks[track_id]["disapeared"] > self.max_disapeared:
                    self._deregister(track_id)
            
        # detections that didnt get matched are given new tracks
        for d_idx, (c,b) in enumerate(zip(input_centroids, input_bboxes)):
            if d_idx not in used_dets:
                self._register(c,b)
        
        return self.tracks
            

def draw_tracks(frame, tracks):
    out = frame.copy()
    for track_id, data in tracks.items():
        x, y, w, h = data["bbox"]
        cx, cy = data["centroid"]
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.circle(out, (int(cx), int(cy)), 3, (0, 0, 255), -1)
        cv2.putText(
            out,
            f"ID {track_id}",
            (x, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )
    return out
            
        