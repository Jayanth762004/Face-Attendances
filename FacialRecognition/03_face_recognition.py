import cv2
import numpy as np
import os
from openpyxl import Workbook
import datetime

recognizer = cv2.face.LBPHFaceRecognizer_create()

# Corrected file path
recognizer.read('E:/OpenCV-Face-Recognition-master/FacialRecognition/trainer/trainer.yml')

cascadePath = "E:/OpenCV-Face-Recognition-master/FacialRecognition/haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)

font = cv2.FONT_HERSHEY_SIMPLEX

# Iniciate id counter
id = 0

# Names related to ids: example ==> Marcelo: id=1, etc
names = ["None ",'jayanth:' , 'Dheeraj', 'vignesh'] 

# Initialize and start realtime video capture
cam = cv2.VideoCapture(0)
cam.set(3, 640) # Set video width
cam.set(4, 480) # Set video height

# Define min window size to be recognized as a face
minW = 0.1 * cam.get(3)
minH = 0.1 * cam.get(4)

# Create a new Excel workbook
wb = Workbook()
ws = wb.active
ws.append(["Name", "Status", "Time"])  # Add header row

face_detected = False

while True:
    ret, img = cam.read()
    img = cv2.flip(img, 1) # Flip vertically

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale( 
        gray,
        scaleFactor = 1.2,
        minNeighbors = 5,
        minSize = (int(minW), int(minH)),
    )

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        id, confidence = recognizer.predict(gray[y:y + h, x: x + w])

        # Check if confidence is less than 100 ==> "0" is a perfect match 
        if confidence < 100:
            id = names[id]
            confidence = "  {0}%".format(round(100 - confidence))
        else:
            id = "unknown"
            confidence = "  {0}%".format(round(100 - confidence))
        
        cv2.putText(img, str(id), (x + 5, y - 5), font, 1, (255, 255, 255), 2)
        cv2.putText(img, str(confidence), (x + 5, y + h - 5), font, 1, (255, 255, 0), 1)  

        if not face_detected:
            ws.append([id, "Entry", str(datetime.datetime.now())])
            face_detected = True

    if len(faces) == 0 and face_detected:
        ws.append([id, "Exit", str(datetime.datetime.now())])
        face_detected = False

    cv2.imshow('camera', img) 

    k = cv2.waitKey(10) & 0xff # Press 'ESC' for exiting video
    if k == 27:
        break

# Save the Excel workbook
wb.save("recognized_names.xlsx")

# Do a bit of cleanup
print("\n [INFO] Exiting Program and cleanup stuff")
cam.release()
cv2.destroyAllWindows()