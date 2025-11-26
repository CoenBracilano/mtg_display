import easyocr
import cv2 

#WIP, currently just runs OCR on a card, needs to be zoomed in enough to be able to read the text though
#need to add logic to figure out what is the name of the card and what is just extra text. 
def parse_card(img):
    reader = easyocr.Reader(['en'])
    result = reader.readtext(img)
    print(result)



if __name__ == "__main__":
    image = cv2.imread()
    parse_card(image)