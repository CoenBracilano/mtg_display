import cv2
import argparse
from card_detection import CardDetector, draw_detections
from tracker import IDTracker, draw_tracks

def main(source):
    # our capturing object, the video feed
    cap = cv2.VideoCapture(source)
    detector = CardDetector()
    tracker = IDTracker(max_disapeared=200, max_distance=80)


    if not cap.isOpened():
        print("error couldnt open")
        return

    # cap.read() processes a new image every frame
    # ret checks if we actualy got something, and frame is what we got
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        detections = detector.detect(frame)
        tracks = tracker.update(detections)
        vis = draw_tracks(frame, tracks)

        cv2.imshow("detections", vis)
        if cv2.waitKey(1) == ord('q'):
            break
        
    # malloc -> free type thing
    cap.release()
    cv2.destroyAllWindows()

# Usage:
# python driver.py --source path_to_video.mp4
# If no source is provided, uses the default webcam (0)
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default=0)
    args = parser.parse_args()
    main(args.source)
    