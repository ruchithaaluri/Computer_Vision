import ultralytics
import PIL
from ultralytics import YOLO
from PIL import Image
import matplotlib.pyplot as plt
import os


model = YOLO("yolov8n.pt") #Load a pretrained YOLOv8n model
image_path = "image.png" #Replace with your image path
if not os.path.exists(image_path):
    raise FileNotFoundError(f"The specified image path does not exist")

#run interference
results = model(image_path)
results = results[0]

#get annotated image(numpy array,BGR) dir
annotated_img = results.plot() #returns a numpy array in BGR format

#convert BGR (open cv style) to RGB for correct color representation in matplotlib
annotated_img_rgb = annotated_img[..., ::-1]

#Display the annotated image
plt.figure(figsize=(10, 8))
plt.imshow(annotated_img_rgb)
plt.axis('off')  # Turn off axis labels
plt.show()

output_path = "output.jpg" 
Image.fromarray(annotated_img_rgb).save(output_path) #Save the annotated image
print(f"Annotated image saved to: {output_path}")

print("Detected results:")
print("-" * 30)

boxes = results.boxes
if len(boxes) == 0:
    print("No objects detected.")
else:
    for i, box in enumerate(boxes, start =1):
        class_id = int(box.cls[0])  # Get the class ID
        class_name = model.names[class_id]  # Get the class name
        confidence = float(box.conf[0])  # Get the confidence score
        x1, y1, x2, y2 = box.xyxy[0].tolist()
       
        print(f"Object {i}:")
        print(f"Class ID: {class_id} ({class_name})")
        print(f"Confidence: {confidence:.2f}")
        print(f"Bounding Box: ({x1:.1f}, {y1:.1f}), ({x2:.1f}, {y2:.1f})")
        print("-" * 30)
        

