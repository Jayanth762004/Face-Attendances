import cv2
import numpy as np
from PIL import Image
import os

# Path for face image database
path = 'E:/OpenCV-Face-Recognition-master/dataset'

# Create trainer directory if it doesn't exist
if not os.path.exists('E:/OpenCV-Face-Recognition-master/FacialRecognition/trainer'):
    os.makedirs('E:/OpenCV-Face-Recognition-master/FacialRecognition/trainer')

recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier("E:/OpenCV-Face-Recognition-master/FacialRecognition/haarcascade_frontalface_default.xml")

# Function to get the images and label data
def getImagesAndLabels(path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    faceSamples = []
    ids = []

    for imagePath in imagePaths:
        try:
            PIL_img = Image.open(imagePath).convert('L')  # convert it to grayscale
        except Exception as e:
            print(f"Error opening image {imagePath}: {e}")
            continue

        img_numpy = np.array(PIL_img, 'uint8')

        try:
            id = int(os.path.split(imagePath)[-1].split(".")[1])
        except ValueError:
            print(f"Error parsing ID from image {imagePath}")
            continue

        faces = detector.detectMultiScale(img_numpy)
        if len(faces) == 0:
            print(f"No faces detected in image {imagePath}")
        
        for (x, y, w, h) in faces:
            faceSamples.append(img_numpy[y:y+h, x:x+w])
            ids.append(id)

    return faceSamples, ids

print("\n[INFO] Training faces. It will take a few seconds. Wait ...")
faces, ids = getImagesAndLabels(path)

if len(faces) == 0 or len(ids) == 0:
    print("[ERROR] No faces or IDs found. Exiting program.")
else:
    recognizer.train(faces, np.array(ids))
    # Save the model into trainer/trainer.yml
    recognizer.write('E:\OpenCV-Face-Recognition-master\FaceDetection\Cascades\trainer.yml')
    # Print the number of faces trained and end program
    print("\n[INFO] {0} faces trained. Exiting Program".format(len(np.unique(ids))))
