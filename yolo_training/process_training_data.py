import io
import os
import numpy as np
from PIL import Image
from pathlib import Path
from datasets import load_dataset

# --- CONFIGURATION ---
TRAIN_COUNT = 5000
VAL_COUNT = 1500
BASE_DIR = Path("mtg_yolo")

def group_corners_by_card(annotations):
    """
    Groups corners into cards using the Corner ID (index 4).
    Expects format: [x, y, visible, angle, corner_id]
    """
    cards = []
    current_card = {}
    
    for corner in annotations:
        # corner[4] is the corner_id (0: TL, 1: TR, 2: BR, 3: BL)
        cid = int(corner[4])
        
        # If we see a 0, we start a new card tracking attempt
        if cid == 0:
            current_card = {0: corner}
        # Only add corners 1, 2, 3 if they belong to the current sequence
        elif cid in [1, 2, 3] and (cid - 1) in current_card:
            current_card[cid] = corner
            
        # Once we have 4 corners in order, we have a valid card
        if len(current_card) == 4:
            cards.append([current_card[0], current_card[1], current_card[2], current_card[3]])
            current_card = {} # Reset for next card
            
    return cards

def corners_to_bbox(card_corners):
    """
    Calculates YOLO center-normalized bbox.
    Clamps coordinates > 1.0 to the image edge.
    """
    # 1. Check visibility (index 2 is the visibility flag)
    visible_count = sum(1 for c in card_corners if c[2] > 0.5)
    
    # NEW: Ignore cards that are too hidden (1 or 0 corners visible)
    if visible_count <= 1:
        return None

    # 2. Extract X (index 0) and Y (index 1)
    xs = [c[0] for c in card_corners]
    ys = [c[1] for c in card_corners]

    # 3. Clamp coordinates to the visible image area [0, 1]
    # This ensures the box stays within the frame even if the card is partially off-screen
    clean_xs = [max(0.0, min(1.0, x)) for x in xs]
    clean_ys = [max(0.0, min(1.0, y)) for y in ys]

    min_x, max_x = min(clean_xs), max(clean_xs)
    min_y, max_y = min(clean_ys), max(clean_ys)

    width = max_x - min_x
    height = max_y - min_y
    
    # Final sanity check for size
    if width < 0.005 or height < 0.005:
        return None

    x_center = min_x + (width / 2)
    y_center = min_y + (height / 2)

    return [x_center, y_center, width, height]

def process_split(stream, split_name):
    print(f"\n🚀 Processing {split_name} split...")
    
    for idx, sample in enumerate(stream):
        if idx % 100 == 0:
            print(f"   Progress: {idx} images saved...")

        # 1. Image Loading (Streaming Fix)
        try:
            raw_img = sample['image']
            if isinstance(raw_img, dict) and 'bytes' in raw_img:
                image = Image.open(io.BytesIO(raw_img['bytes']))
            elif isinstance(raw_img, bytes):
                image = Image.open(io.BytesIO(raw_img))
            else:
                image = raw_img # PIL object
            
            if image.mode != 'RGB':
                image = image.convert('RGB')
        except Exception as e:
            print(f"   ⚠️ Skipping img_{idx}: {e}")
            continue

        # Save Image (Native Res, Quality 85 for space)
        img_filename = f"img_{idx:06d}.jpg"
        img_path = BASE_DIR / 'images' / split_name / img_filename
        image.save(img_path, "JPEG", quality=85, optimize=True)

        # 2. Annotation Processing
        # annotations is a list of [x, y, visible, angle, corner_id]
        annotations = sample['annotation']
        cards = group_corners_by_card(annotations)
        
        labels = []
        for card_corners in cards:
            bbox = corners_to_bbox(card_corners)
            if bbox:
                # Format to 6 decimal places for YOLO precision
                labels.append(f"0 {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}")

        # Save Label File
        label_filename = f"img_{idx:06d}.txt"
        label_path = BASE_DIR / 'labels' / split_name / label_filename
        with open(label_path, 'w') as f:
            f.write('\n'.join(labels))

def main():
    # Setup folders
    for split in ['train', 'val']:
        (BASE_DIR / 'images' / split).mkdir(parents=True, exist_ok=True)
        (BASE_DIR / 'labels' / split).mkdir(parents=True, exist_ok=True)

    # Connect to HF stream
    print("📡 Connecting to Hugging Face Stream...")
    train_stream = load_dataset("gabraken/mtg-detection", split="train", streaming=True).take(TRAIN_COUNT)
    val_stream = load_dataset("gabraken/mtg-detection", split="test", streaming=True).take(VAL_COUNT)

    process_split(train_stream, 'train')
    process_split(val_stream, 'val')

    # Create YAML
    yaml_content = f"path: {BASE_DIR.absolute()}\ntrain: images/train\nval: images/val\nnc: 1\nnames: ['card']"
    with open(BASE_DIR / "data.yaml", "w") as f:
        f.write(yaml_content)

    print("\n✅ Dataset conversion complete at native resolution.")

if __name__ == "__main__":
    main()