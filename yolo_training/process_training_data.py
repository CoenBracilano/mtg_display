from datasets import load_dataset
from pathlib import Path
import numpy as np
from PIL import Image
import io

# Load dataset
print("Loading dataset from Hugging Face...")
dataset = load_dataset("gabraken/mtg-detection")

# Create YOLO directory structure
base_path = Path("mtg_yolo")
for split in ['train', 'val']:
    (base_path / 'images' / split).mkdir(parents=True, exist_ok=True)
    (base_path / 'labels' / split).mkdir(parents=True, exist_ok=True)

def find_next_corner(current_corner, all_corners, visited):
    """Find the next corner by following the angle vector"""
    # current_corner format: [x, y, visible, angle, corner_id]
    curr_x, curr_y, _, angle, _ = current_corner
    
    # Direction vector from angle
    dx = np.cos(angle)
    dy = np.sin(angle)
    
    best_match = None
    best_score = float('inf')
    
    for idx, corner in enumerate(all_corners):
        if idx in visited:
            continue
        
        # corner format: [x, y, visible, angle, corner_id]
        to_x = corner[0] - curr_x
        to_y = corner[1] - curr_y
        
        # Distance
        dist = np.sqrt(to_x**2 + to_y**2)
        if dist < 0.01:  # Too close
            continue
        
        # Normalize
        to_x /= dist
        to_y /= dist
        
        # Check alignment with angle direction
        alignment = dx * to_x + dy * to_y
        
        # Score: prefer aligned and close corners
        score = dist * (2 - alignment)
        
        if alignment > 0.5 and score < best_score:
            best_score = score
            best_match = {'corner': corner, 'idx': idx}
    
    return best_match

def group_corners_by_card(corners):
    """Group corners into cards by following the angle chain"""
    if not corners:
        return []
    
    visited = set()
    cards = []
    
    for start_idx, start_corner in enumerate(corners):
        if start_idx in visited:
            continue
        
        # Start a new card
        card_corners = [start_corner]
        visited.add(start_idx)
        current = start_corner
        
        # Follow the angle chain to find next 3 corners
        for _ in range(3):
            next_corner = find_next_corner(current, corners, visited)
            if next_corner is None:
                break
            
            card_corners.append(next_corner['corner'])
            visited.add(next_corner['idx'])
            current = next_corner['corner']
        
        # Valid card has 4 corners
        if len(card_corners) == 4:
            cards.append(card_corners)
    
    return cards

def corners_to_bbox(card_corners):
    """Convert 4 corners to YOLO bbox format (full extent)"""
    # card_corners format: list of [x, y, visible, angle, corner_id]
    
    # Check visibility - need at least 2 visible corners
    visible_count = sum(1 for corner in card_corners if corner[2] > 0.5)
    if visible_count < 2:
        return None
    
    # Extract all coordinates (clip to 0-1 range for visible area)
    coords = []
    for corner in card_corners:
        x, y = corner[0], corner[1]
        # Clip coordinates to image bounds [0, 1]
        x = max(0.0, min(1.0, x))
        y = max(0.0, min(1.0, y))
        coords.append((x, y))
    
    xs = [c[0] for c in coords]
    ys = [c[1] for c in coords]
    
    x_center = (min(xs) + max(xs)) / 2
    y_center = (min(ys) + max(ys)) / 2
    width = max(xs) - min(xs)
    height = max(ys) - min(ys)
    
    # Sanity checks
    if width < 0.05 or height < 0.05:  # Too small
        return None
    if width > 0.95 or height > 0.95:  # Suspiciously large
        return None
    
    return [x_center, y_center, width, height]

def process_split(dataset_split, split_name):
    """Process train or test split"""
    print(f"\nProcessing {split_name} split ({len(dataset_split)} images)...")
    
    for idx, sample in enumerate(dataset_split):
        if idx % 1000 == 0:
            print(f"  Progress: {idx}/{len(dataset_split)} images...")
        
        # Convert bytes to PIL Image
        image_data = sample['image']
        
        if isinstance(image_data, bytes):
            image = Image.open(io.BytesIO(image_data))
        elif isinstance(image_data, dict) and 'bytes' in image_data:
            image = Image.open(io.BytesIO(image_data['bytes']))
        else:
            image = image_data
        
        # Convert to PIL if needed
        if not isinstance(image, Image.Image):
            image = Image.fromarray(np.array(image))
        
        # Convert RGBA to RGB (fix for JPEG)
        if image.mode == 'RGBA':
            rgb_image = Image.new('RGB', image.size, (255, 255, 255))
            rgb_image.paste(image, mask=image.split()[3])
            image = rgb_image
        elif image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Save image
        img_path = base_path / 'images' / split_name / f'img_{idx:06d}.jpg'
        image.save(img_path)
        
        # Process corners
        corners = sample['annotation']
        
        # Group corners into cards
        cards = group_corners_by_card(corners)
        
        # Convert to YOLO bboxes
        labels = []
        for card_corners in cards:
            bbox = corners_to_bbox(card_corners)
            if bbox:
                labels.append(f"0 {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}")
        
        # Save labels
        label_path = base_path / 'labels' / split_name / f'img_{idx:06d}.txt'
        with open(label_path, 'w') as f:
            f.write('\n'.join(labels))
    
    print(f"  ✓ Completed {split_name}: {len(dataset_split)} images processed")

# Process both splits
process_split(dataset['train'], 'train')
process_split(dataset['test'], 'val')

# Create data.yaml
yaml_content = f"""path: {base_path.absolute()}
train: images/train
val: images/val

nc: 1
names: ['card']
"""

yaml_path = base_path / 'data.yaml'
with open(yaml_path, 'w') as f:
    f.write(yaml_content)

print("\n" + "="*60)
print("✅ Conversion complete!")
print(f"📁 Dataset location: {base_path.absolute()}")
print(f"📄 Config file: {yaml_path.absolute()}")
print("\n🚀 To train YOLOv8:")
print(f"   yolo detect train data={yaml_path} model=yolov8n.pt epochs=50 imgsz=1024 batch=8")
print("="*60)