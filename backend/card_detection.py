import cv2
import numpy as np
import os

class CardDetector:
    def __init__(self, min_area=1000, max_area= 200000, epsilon_factor = 0.02) :
        self.min_area = min_area
        self.max_area = max_area
        self.epsilon_factor = epsilon_factor

    def detect(self, frame):
        # Convert to greyscale and blur to remove background, 5x5 gaussian kernel
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5,5), 0)
        
        # Canny edge detection, really good at determining between strong edges
        # Such as the edge of a card and a darker background, less robust for unclear lines
        # but very strong and fast for good lines
        edges = cv2.Canny(blur, 50,150)
        # make our edges more defined
        edges = cv2.dilate(edges, None, iterations=1)

        # contour = closed curve of boundry for an object
        # findContours gives us the list of candidate objects that are closed and bounded
        # RETER_EXTERNAL - only the outermost boundry, not internal things like card art
        # CHAIN_APPROX_SIMPLE - instead of storing every point on the curve, do an approximation
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detections = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            # cards are a fixed size so once the camera is mounted, we can filter out objects which are too small or large
            # this can be a little finiky to fine tune  
            if area < self.min_area or area > self.max_area:
                continue
            
            peri = cv2.arcLength(cnt, True) # find perimiter
            approx = cv2.approxPolyDP(cnt, self.epsilon_factor * peri, True)
            # the above runs the Douglas-Peuker algorithm, which approximates curves or contours with fewer points
            # for us, it takes the exact rounded form of a card and tries to match it to a rectangle of similar shape
            # the epsilon factor is how close the approimation can be, we will keep this low because the difference between a rectangle
            # and a card should be very close. Currently at 2% of the contors perimiter

            #goal is to find quadrelaterals which are convex, so if 4 points or if convex
            if len(approx) != 4 or not cv2.isContourConvex(approx):
                continue
            
            # use our succesful approximation to make a bounding box (rectangle)
            # the w,h check should never really happen its just easy and a good piece of mind
            x,y,w,h = cv2.boundingRect(approx)
            if w == 0 or h == 0:
                continue
            # find the center of the card
            cx = x+w / 2.0
            cy = y+h / 2.0
            # store
            detections.append({
                "bbox": (x,y,w,h),
                "centroid": (cx, cy),
                "contour": (approx)
            })
        
        return detections

# take in a frame and some detected objcts
# draw a box around the outside and a dot at the center
def draw_detections(frame, detections):
    out = frame.copy()
    for det in detections:
        x,y,w,h = det["bbox"]
        cx, cy = det["centroid"]
        cv2.rectangle(out, (x,y), (x+w, y+h), (0,255,0), 2)
        cv2.circle(out, (int(cx), int(cy)), 3, (0, 0, 255), -1)
    return out

# Warps any detected cards for easier feature matching
# Saves warped cards to ../detected_cards/ for debugging
def warp_and_save(frame, detections, idx=0, width=300, height=420):
    if os.path.exists("../detected_cards"):
        for filename in os.listdir("../detected_cards"):
            file_path = os.path.join("../detected_cards", filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
        os.rmdir("../detected_cards")
    os.makedirs("../detected_cards")

    warped_cards = []
    for idx, detection in enumerate(detections):
        pts = detection["contour"].reshape(4, 2)
        pts = order_points(pts)

        dst = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype="float32")
        M = cv2.getPerspectiveTransform(pts.astype("float32"), dst)
        warped = cv2.warpPerspective(frame, M, (width, height))
        warped_cards.append(warped)

        cv2.imwrite(f"../detected_cards/card_{idx+1}.jpg", warped)            

    return warped_cards

# Helper function to order points for perspective transform
def order_points(pts):
        rect = np.zeros((4, 2), dtype="float32")

        s = np.sum(pts, axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]

        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]

        return rect