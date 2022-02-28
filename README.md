# LicencePlateRecognition
Yolov5 Licence Plate Recognition

This is a yolov5 license plate recognition project. After detecting the plate in the image, the plate was read using Tesseract OCR. 

Requirements
- YoloV5
- Tesseract OCR
- OpenCV

Pretrained Model
- https://drive.google.com/file/d/122K-LzNBlqnjbarHH3ztx1Ol6uyohVcF/view?usp=sharing

Using
1- conda create --name LicencePlate python=3.7 anaconda
2- git clone https://github.com/ultralytics/yolov5
3- cd yolov5
4- pip install -r requirements.txt
5- pip install pytesseract 
6- pip install imutils
7- python yolo_pretrained_image.py

Test Environment
- Ubuntu 20.04
- Python 3.7.11
- OpenCv 4.5.5
- Numpy 1.17.3
- Pytesseract 0.3.9
