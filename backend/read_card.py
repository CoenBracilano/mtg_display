import easyocr
import cv2
import argparse

#WIP, currently just runs OCR on a card, needs to be zoomed in enough to be able to read the text though
#need to add logic to figure out what is the name of the card and what is just extra text. 
def parse_card(img):
    reader = easyocr.Reader(['en'])
    result = reader.readtext(img)
    for res in result:
        print(res[1])


# Usage:
# python read_card.py --file path_to_image.jpg
# If no file is provided, uses "../test_files/mtgcards.webp"
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, default="../test_files/mtgcards.webp")
    args = parser.parse_args()
    image = cv2.imread(args.file)

    parse_card(image)