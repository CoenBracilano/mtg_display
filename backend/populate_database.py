import os
import cv2
import numpy as np
import requests
import json

# Populate the card database from a list of card names
def populate(db_path, card_list_path):

    if not os.path.exists(db_path):
        os.makedirs(db_path)

    if not os.path.exists(card_list_path):
        print("Card list file does not exist")
        return    
    
    # Grab list of card names to populate db with from file
    file = open(card_list_path, 'r', encoding='utf-8')
    names = file.readlines()
    file.close()
    card_names = [name.strip() for name in names if name.strip()]
    
    # Initialize ORB detector and card storage
    orb = cv2.ORB_create(nfeatures=500)    
    cards = {}

    for name in card_names:
        # Grab card data from Scryfall 
        url = f"https://api.scryfall.com/cards/named?exact={requests.utils.quote(name)}"
        response = requests.get(url)
        card = response.json()
        img_url = card.get('image_uris', {}).get('normal')
        
        # Download card image
        img_data = requests.get(img_url).content
        img_path = os.path.join(db_path, f"{name}.jpg")
        file = open(img_path, 'wb')
        file.write(img_data)
        file.close()
        
        # Feature detection of card image
        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = orb.detectAndCompute(gray, None)
        if descriptors is not None:
            cards[name] = {
                'descriptors': descriptors,
                'num_features': len(keypoints),
                'url': card.get('image_uris', {}).get('normal', ''),
            }
    
    # Save database card features to JSON
    db_file = os.path.join(db_path, "cards.json")
    file = open(db_file, 'w')
    
    cards_formatted = {}
    for name, data in cards.items():
        cards_formatted[name] = {
            'descriptors': data['descriptors'].tolist(),
            'num_features': data['num_features'],
            'url': data.get('url', '')
        }
    
    json.dump(cards_formatted, file)
    file.close()
    print("Database population complete")
      