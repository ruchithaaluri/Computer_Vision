IMAGE_PATH = "ocr.png"  # Replace with your image path
def run_easyocr(IMAGE_PATH):
    import easyocr
    reader = easyocr.Reader(['en'],gpu=False)
    results = reader.readtext(IMAGE_PATH)
    for (bbox, text, prob) in results:
        print(f"Text: {text!r:30} prob={prob:.3f} bbox={bbox}") 
        
if __name__ == "__main__" :
        run_easyocr(IMAGE_PATH) 
        