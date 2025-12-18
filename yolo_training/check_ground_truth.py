import sys
import os
from pathlib import Path
import cv2
import numpy as np

# 1. Get the path to the folder where THIS script is (yolo_training)
current_script_path = Path(__file__).resolve()

# 2. Get the path to the 'mtg_display' folder (the parent)
project_root = current_script_path.parent.parent

# 3. Add the project root to Python's search list
sys.path.append(str(project_root))

# 4. Now import using the folder name 'backend'
try:
    from backend.card_detection import CardDetector
    print("Successfully imported CardDetector")
except ImportError as e:
    print(f"Import failed. Checked path: {project_root}")
    print(f"Error: {e}")

def compare_detections(image_dir, label_dir):
    # Initialize your detector
    detector = CardDetector()
    
    img_path = Path(image_dir)
    lbl_path = Path(label_dir)
    
    valid_extensions = ('.jpg', '.jpeg', '.png')
    image_files = [f for f in img_path.iterdir() if f.suffix.lower() in valid_extensions]

    print(f"Comparing {len(image_files)} images.")
    print("RED = Ground Truth (Label File)")
    print("GREEN = Your Algorithm (CardDetector)")
    print("Press 'q' to quit, any other key for next image.")

    for img_file in image_files:
        # 1. Load Image
        img = cv2.imread(str(img_file))
        if img is None: continue
        h, w, _ = img.shape
        
        # --- DRAW GROUND TRUTH (RED) ---
        lbl_file = lbl_path / (img_file.stem + ".txt")
        if lbl_file.exists():
            with open(lbl_file, 'r') as f:
                for line in f.readlines():
                    parts = line.strip().split()
                    if len(parts) == 5:
                        _, xc, yc, bw, bh = map(float, parts)
                        # Convert YOLO to pixels
                        x1 = int((xc - bw/2) * w)
                        y1 = int((yc - bh/2) * h)
                        x2 = int((xc + bw/2) * w)
                        y2 = int((yc + bh/2) * h)
                        # Draw Red Rectangle
                        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        cv2.putText(img, "GT", (x1, y1 - 25), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # --- DRAW YOUR ALGORITHM DETECTIONS (GREEN) ---
        # This uses the 'detect' method from your CardDetector class
        detections = detector.detect(img)
        for det in detections:
            x, y, bw, bh = det["bbox"]
            # Draw Green Rectangle
            cv2.rectangle(img, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
            cv2.putText(img, "ALGO", (x, y - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Display Result
        cv2.imshow("Comparison: Red(GT) vs Green(Algo)", img)
        
        key = cv2.waitKey(0)
        if key == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    # --- UPDATE THESE PATHS ---
    IMAGE_FOLDER = "mtg_yolo/images/train"
    LABEL_FOLDER = "mtg_yolo/labels/train"
    
    compare_detections(IMAGE_FOLDER, LABEL_FOLDER)