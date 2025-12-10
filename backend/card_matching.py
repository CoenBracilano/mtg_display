import argparse
import cv2
import os
import numpy as np
import requests
import json
from card_detection import CardDetector, warp_and_save

class CardMatcher:
    def __init__(self, db_path):
        
        self.db_path = db_path

        # ORB detector (fast and patent-free)
        self.orb = cv2.ORB_create(nfeatures=500)
        
        # BFMatcher for matching features
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        
        # Card database: {card_name: {'descriptors': ..., 'keypoints': ...}}
        self.cards = {}
        self.load()

    # Identify all cards in the given image
    # def identify_all_cards(self, img_path, min_matches=15, dist_threshold=50):

    #     img = cv2.imread(img_path)
    #     if img is None:
    #         print(f"Error: Could not read image")
    #         return None

    #     # Detect cards in the image
    #     detections = self.detector.detect(img)
    #     if len(detections) == 0:
    #         print("No cards detected in image")
    #         return []
        
    #     # Warp detected cards for better matching
    #     warped_cards = warp_and_save(img, detections)

    #     # Identify each detected card
    #     results = []
    #     for card_img in warped_cards:
    #         match = self.identify(card_img, min_matches, dist_threshold)
    #         results.append({
    #             "image": card_img,
    #             "match": match
    #         })

    #     return results
    
    # Identify the card in the given transformed card image
    def identify(self, card_img, min_matches=15, dist_threshold=50):

        # Feature detection of input image
        img = card_img.copy()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = self.orb.detectAndCompute(gray, None)

        if descriptors is None:
            print("No features detected in image")
            return None
        
        # Match against all cards in database
        best_match = None
        best_num_matches = 0
        for name, data in self.cards.items():
            card_descriptors = data['descriptors']
                
            # Match features
            matches = self.matcher.match(descriptors, card_descriptors)
            
            # Filter good matches
            good_matches = [match for match in matches if match.distance < dist_threshold]
            
            if len(good_matches) == 0:
                continue
            
            # Calculate match quality
            num_matches = len(good_matches)
            avg_distance = np.mean([match.distance for match in good_matches])   
            
            # Confidence score based on number of matches and quality
            confidence = min(100, (num_matches / min_matches) * 100 * (1 - avg_distance/100))
            
            # Keep track of best match
            if num_matches > best_num_matches:
                best_num_matches = num_matches
                best_match = {
                    'name': name,
                    'num_matches': num_matches,
                    'avg_distance': float(avg_distance),
                    'confidence': max(0, float(confidence)),
                    'url': data.get('url', '')
                }
                
        # Return best match if its valid enough    
        if best_match and best_match['num_matches'] >= min_matches / 2:
            return best_match
        
        return None
        
    # Load database card features from JSON
    def load(self):
        db_file = os.path.join(self.db_path, "cards.json")
        
        if os.path.exists(db_file):
            file = open(db_file, 'r')
            db = json.load(file)
            file.close()

            self.cards = {}
            for name, data in db.items():
                self.cards[name] = {
                    'descriptors': np.array(data['descriptors'], dtype=np.uint8),
                    'num_features': data['num_features'],
                    'url': data.get('url', '')
                }
        else:
            print("No existing database found")


# # Usage:
# # python card_matching.py --card_list path_to_card_list.txt --image path_to_image.jpg
# # If no card list is provided, uses "../test_files/mtgcards_card_list.txt"
# # If no image is provided, uses "../test_files/mtgcards.webp"
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--card_list", type=str, default="../test_files/mtgcards_card_list.txt")
#     parser.add_argument("--image", type=str, default="../test_files/mtgcards.webp")
#     args = parser.parse_args()

#     matcher = CardMatcher()
#     matcher.populate(args.card_list)
 
#     results = matcher.identify_all_cards(args.image)
#     for idx, card in enumerate(results):
#         print(f"\nCard {idx+1}:")

#         candidates = card["match"]
#         if not candidates:
#             print("No good match found")
#             continue

#         # Best match
#         best = candidates[0]
#         print(f"Best Match: {best['name']}")
#         print(f"Matches: {best['num_matches']}")
#         print(f"Confidence: {best['confidence']:.0f}%")

#         # Other candidates
#         if len(candidates) > 1:
#             print("Other candidates:")
#             for candidate in candidates[1:]:
#                 print(f"- {candidate['name']} ({candidate['num_matches']} matches)")

