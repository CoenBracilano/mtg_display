import os
import cv2
import numpy as np
import requests
import json
import argparse
import time
from urllib.parse import quote

def get_all_prints(card_name):
    """
    Get all printings/art variations of a card from Scryfall.
    
    Returns:
        List of card objects with different arts
    """
    # Search for all printings of the card
    search_url = f"https://api.scryfall.com/cards/search?q=!'{quote(card_name)}'+unique:prints"
    
    all_cards = []
    
    try:
        while search_url:
            response = requests.get(search_url)
            
            if response.status_code == 429:  # Rate limited
                print("Rate limited, waiting 100ms...")
                time.sleep(0.1)
                continue
            
            if response.status_code != 200:
                print(f"Error fetching {card_name}: {response.status_code}")
                break
            
            data = response.json()
            
            # Filter cards that have images
            cards_with_images = [
                card for card in data.get('data', [])
                if card.get('image_uris') or card.get('card_faces', [{}])[0].get('image_uris')
            ]
            
            all_cards.extend(cards_with_images)
            
            # Check for pagination
            search_url = data.get('next_page')
            
            if search_url:
                time.sleep(0.1)  # Respect Scryfall rate limits
        
        return all_cards
    
    except Exception as e:
        print(f"Error fetching prints for {card_name}: {e}")
        return []

def get_card_image_url(card):
    """Extract image URL from card object (handles double-faced cards)."""
    if card.get('image_uris'):
        return card['image_uris'].get('normal')
    elif card.get('card_faces'):
        # For double-faced cards, use the front face
        return card['card_faces'][0].get('image_uris', {}).get('normal')
    return None

def populate(db_path, card_list_path, include_all_arts=True):
    """
    Populate the card database from a list of card names.
    
    Args:
        db_path: Path to store the database
        card_list_path: Path to text file with card names
        include_all_arts: If True, include all art variations
    """
    print("Populating card database...")
    
    if not os.path.exists(db_path):
        os.makedirs(db_path)
    
    if not os.path.exists(card_list_path):
        print("Card list file does not exist")
        return
    
    # Read card names from file
    with open(card_list_path, 'r', encoding='utf-8') as f:
        card_names = [name.strip() for name in f.readlines() if name.strip()]
    
    print(f"Found {len(card_names)} unique card names")
    
    # Initialize ORB detector
    orb = cv2.ORB_create(nfeatures=500)
    cards = {}
    total_variations = 0
    
    for idx, card_name in enumerate(card_names, 1):
        print(f"\n[{idx}/{len(card_names)}] Processing: {card_name}")
        
        if include_all_arts:
            # Get all printings/arts
            printings = get_all_prints(card_name)
            print(f"  Found {len(printings)} art variations")
        else:
            # Get only the default printing
            try:
                url = f"https://api.scryfall.com/cards/named?exact={quote(card_name)}"
                response = requests.get(url)
                if response.status_code == 200:
                    printings = [response.json()]
                else:
                    printings = []
            except Exception as e:
                print(f"  Error: {e}")
                printings = []
        
        # Process each art variation
        for print_idx, card_data in enumerate(printings):
            # Get image URL
            img_url = get_card_image_url(card_data)
            
            if not img_url:
                continue
            
            # Create unique name for this variation
            set_code = card_data.get('set', 'unknown').upper()
            collector_number = card_data.get('collector_number', '0')
            
            # Unique identifier: CardName_SET_CollectorNumber
            unique_name = f"{card_name}_{set_code}_{collector_number}"
            
            # Skip if already processed
            if unique_name in cards:
                continue
            
            try:
                # Download image
                img_response = requests.get(img_url, timeout=10)
                
                if img_response.status_code != 200:
                    print(f"  Failed to download {set_code} #{collector_number}")
                    continue
                
                # Save image to disk
                img_path = os.path.join(db_path, f"{unique_name}.jpg")
                with open(img_path, 'wb') as f:
                    f.write(img_response.content)
                
                # Load and process image
                img = cv2.imread(img_path)
                
                if img is None:
                    print(f"  Failed to load image for {unique_name}")
                    os.remove(img_path)
                    continue
                
                # Extract features
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                keypoints, descriptors = orb.detectAndCompute(gray, None)
                
                if descriptors is not None and len(keypoints) > 0:
                    cards[unique_name] = {
                        'descriptors': descriptors,
                        'num_features': len(keypoints),
                        'url': img_url,
                        'card_name': card_name,  # Original card name
                        'set': set_code,
                        'collector_number': collector_number
                    }
                    total_variations += 1
                    print(f"  Added {set_code} #{collector_number} ({len(keypoints)} features)")
                else:
                    print(f"  No features found for {unique_name}")
                    os.remove(img_path)
                
                # Rate limiting
                time.sleep(0.075)  # 75ms between requests (~13 req/sec, well under Scryfall's limit)
                
            except Exception as e:
                print(f"  Error processing {unique_name}: {e}")
                continue
    
    # Save database
    print(f"\n{'='*60}")
    print(f"Saving database with {total_variations} card variations...")
    
    db_file = os.path.join(db_path, "cards.json")
    
    # Convert numpy arrays to lists for JSON serialization
    cards_formatted = {}
    for unique_name, data in cards.items():
        cards_formatted[unique_name] = {
            'descriptors': data['descriptors'].tolist(),
            'num_features': data['num_features'],
            'url': data['url'],
            'card_name': data.get('card_name', ''),
            'set': data.get('set', ''),
            'collector_number': data.get('collector_number', '')
        }
    
    with open(db_file, 'w') as f:
        json.dump(cards_formatted, f, indent=2)
    
    print(f"Database saved to {db_file}")
    print(f"Statistics:")
    print(f"   - Unique card names: {len(card_names)}")
    print(f"   - Total variations: {total_variations}")
    print(f"   - Average variations per card: {total_variations/len(card_names):.1f}")
    print(f"{'='*60}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Populate card database from list of card names with all art variations."
    )
    parser.add_argument(
        "--db_path", 
        type=str, 
        default="../card_database", 
        help="Path to store the card database."
    )
    parser.add_argument(
        "--card_list", 
        type=str, 
        default="../test_files/urza_deck_list.txt", 
        help="Path to text file containing card names (one per line)."
    )
    parser.add_argument(
        "--single-art",
        action="store_true",
        help="Only fetch one art per card (default art) instead of all variations."
    )
    
    args = parser.parse_args()
    
    populate(
        db_path=args.db_path,
        card_list_path=args.card_list,
        include_all_arts=not args.single_art
    )