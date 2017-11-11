import cv2
import numpy as np


facedetect = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml');
eyesdetect = cv2.CascadeClassifier('haarcascade_eye.xml');
cam = cv2.VideoCapture(0);

while(True):
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face = facedetect.detectMultiScale(gray, 1.3, 5);
    eyes = eyesdetect.detectMultiScale(gray, 1.3, 5);
    # To make a rectangle across face
    for (x, y, w, h) in face:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    # To make a rectangle along eyes
    for (ex, ey, ew, eh) in eyes:
        cv2.rectangle(img, (ex, ey), (ex+ew, ey+eh), (0, 0, 255), 2) 
    
    # To show the final image 
    cv2.imshow('face', img);
    if(cv2.waitKey(1)==ord('q')):
        break;

cv2.release()
cv2.destroyAllWindows()