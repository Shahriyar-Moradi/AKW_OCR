import cv2
import easyocr
import matplotlib.pyplot as plt
import json
import numpy as np
import os

# Initialize the EasyOCR Reader
reader = easyocr.Reader(['en'])

IMAGE_PATH = 'images/Kyrgyzstan.jpg'

img = cv2.imread(IMAGE_PATH)
if img is None:
    raise ValueError("Image not found or unable to read the image.")
# Function to enhance low-contrast images
def enhance_contrast(image):
    # Convert image to LAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))

    # Convert back to BGR color space
    enhanced_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return enhanced_img

# Function to save JSON output for each image
def save_json_output(image_path, detections):
    json_output = {
        'image_path': image_path,
        'recognized_text': [
            {
                'Detected text': detection['text'],
                'Bounding box': detection['bounding_box']
            } for detection in detections
        ]
    }
    json_filename = image_path.rsplit('.', 1)[0] + '.json'

    json_output_converted = {
        'image_path': json_output['image_path'],
        'recognized_texts': [
            {
                'Detected text': d['Detected text'],
                'Bounding box': [list(map(int, point)) for point in d['Bounding box']]
            } for d in json_output['recognized_text']
        ]
    }

    with open(json_filename, 'w') as json_file:
        json.dump(json_output_converted, json_file, indent=4)

# Function to manually verify the accuracy of the recognized text
def manual_verification(image_path, detections):
    print(f"Manual verification for image: {image_path}")
    for i, detection in enumerate(detections):
        bounding_box = detection['bounding_box']
        text = detection['text']
        confidence = detection['confidence']
        
        print(f"\nDetection {i + 1}:")
        print(f"Detected text: {text}")
        print(f"Bounding box: {bounding_box}")
        print(f"Confidence: {confidence}")
        
        # actual_text = input("Enter the actual text (or press Enter to accept the detected text): ").strip()
        
        # if actual_text:
        #     detection['text'] = actual_text
        
        print(f"Final text for detection {i + 1}: {detection['text']}")

# Function to save the processed image with bounding boxes
def save_processed_image(image_path, img):
    processed_image_path = image_path.rsplit('.', 1)[0] + '_processed.jpg'
    cv2.imwrite(processed_image_path, img)
    print(f"Processed image saved at: {processed_image_path}")

# Read the image


# Enhance the image for better OCR accuracy
img = enhance_contrast(img)

# Perform OCR
result = reader.readtext(img)

# List to collect OCR results
detections = []

# Iterate through detections
for detection in result:
    top_left = tuple(map(int, detection[0][0]))
    bottom_right = tuple(map(int, detection[0][2]))
    text = detection[1]
    confidence = detection[2]
    
    # Draw rectangle and put text
    img = cv2.rectangle(img, top_left, bottom_right, (0, 255, 0), 3)
    
    # Collect the results
    detections.append({
        'bounding_box': detection[0],
        'text': text,
        'confidence': confidence
    })

# Save the processed image with bounding boxes
save_processed_image(IMAGE_PATH, img)

# Convert BGR image to RGB for displaying with matplotlib
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img_rgb)
plt.axis('off')  # Hide axes
plt.show()

# Function to calculate accuracy 
def calculate_accuracy(ground_truth, recognized_text):
    gt_words = ground_truth.split()
    rec_words = recognized_text.split()
    correct = sum(1 for gt, rec in zip(gt_words, rec_words) if gt == rec)
    accuracy = (correct / len(gt_words)) * 100 if gt_words else 0
    return accuracy

# Save the JSON output
save_json_output(IMAGE_PATH, detections)