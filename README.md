# AKW_OCR
Optical Character Recognition or OCR, is a technology that enables you to convert different types of documents, such as scanned paper documents, PDF files or images captured by a digital camera into editable and searchable data.
They are some algorithms that are used to recognize text from images like Tesseract,Keras-OCR,EasyOCR.
So I have used EasyOCR to recognize text from images since it is very easy to use and it has a good accuracy becuase I worked with it and other libaries.

# A brief description of the approach taken:
first I Initialize the EasyOCR Reader, then I enhance the image Contrast by Limited Adaptive Histogram Equalization.
Then I pass the image to the reader and get the text from the image and detect the results from the reader that contains the text,bounding boxes,confidence scores.After that I draw the bounding boxes and save the image with bounding boxes and save text and bounding boxes in a JSON file.
Finally I wrote a function named calculate_accuracy for calculating the accuracy of the model based on the ground truth and detected results that it needs to call with the ground truth and detected results.



# Instructions on how to run the code: 

create a virtual environment using the following command:
python -m venv venv .
actiave venv with the following command: 
source venv/bin/activate .


Install the requirements using the following command: 
pip install -r requirements.txt

