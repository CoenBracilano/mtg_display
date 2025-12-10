import cv2
import argparse
from card_matching import CardMatcher
from frame_processor import FrameProcessor
from populate_database import populate

def main(source, card_list):

    cap = cv2.VideoCapture(source)
    processor = FrameProcessor()

    if not cap.isOpened():
        print("error couldnt open")
        return
    
    # Populate database from card list
    # Card database will be created in "../card_database" (or whatever if we want to change it later)
    populate("../card_database", card_list)
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Process frame
        output = processor.process_frame(frame)
        
        # Add overlay with stats
        stats = processor.get_stats()
        overlay_text = [
            f"Frame: {frame_count}",
            f"Tracked: {stats['active_tracks']}",
            f"Identified: {stats['identified_cards']}",
            f"Unique: {stats['unique_cards']}"
        ]
        
        y_pos = 30
        for text in overlay_text:
            cv2.putText(output, text, (10, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            y_pos += 25
        
        # Display
        cv2.imshow('MTG Card Recognition', output)
        
        # Check for quit
        if cv2.waitKey(1) == ord('q'):
            break
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    
    # Final stats
    stats = processor.get_stats()
    print("\n" + "="*60)
    print("SESSION SUMMARY")
    print("="*60)
    print(f"Frames processed: {frame_count}")
    print(f"Cards in database: {stats['database_size']}")
    print(f"Unique cards seen: {stats['unique_cards']}")
    print(f"Total identifications: {stats['identified_cards']}")
    print("="*60 + "\n")

# Usage:
# python matching_driver.py --source path_to_video.mp4 --card_list path_to_card_list
# If no source is provided, uses the default webcam (0)
# If no card list is provided, uses "../test_files/mtgcards_card_list.txt"
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default="0")
    parser.add_argument("--card_list", type=str, default="../test_files/mtgcards_card_list.txt")
    args = parser.parse_args()
    
    # Convert source to int for webcams
    if (args.source.isdigit()):
        source = int(args.source)
    else:
        source = args.source

    main(source, args.card_list)