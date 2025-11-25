import cv2
from card_detection import CardDetector, draw_detections

def main():
    # our capturing object, the video feed
    cap = cv2.VideoCapture(0)
    detector = CardDetector()

    if not cap.isOpened():
        print("error couldnt open")
        return

    #cap.read() processes a new image every frame
    # ret checks if we actualy got something, and frame is what we got
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        detections = detector.detect(frame)
        vis = draw_detections(frame, detections)

        cv2.imshow("detections", vis)
        if cv2.waitKey(1) == ord('q'):
            break
        
    # malloc -> free type thing
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
    