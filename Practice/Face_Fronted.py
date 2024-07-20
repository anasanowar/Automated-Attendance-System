
# Face Recognition

# Importing the libraries
import PIL
from keras.applications.vgg16 import preprocess_input
import base64
from io import BytesIO
import json
import random
import cv2
from keras.models import load_model
import numpy as np

from keras.preprocessing import image

import csv
from collections import deque

# Initialize variables for tracking consecutive detections
consecutive_frames = 0
max_consecutive_frames = 10  # Number of consecutive frames required for action

# Initialize a deque to track recent detected names
detected_names = deque(maxlen=max_consecutive_frames)

# CSV filename for writing detected names
csv_filename = 'C:/Users/ASUS/Downloads/detected_names.csv'

# Open CSV file in append mode to add new entries
csv_file = open(csv_filename, 'a', newline='')

# CSV writer object
csv_writer = csv.writer(csv_file)

model = load_model('C:/Users/ASUS/Downloads/facefeatures_new_model_1.h5')

# Loading the cascades
face_cascade = cv2.CascadeClassifier('C:/Users/ASUS/Downloads/haarcascade_frontalface_default.xml')


def face_extractor(img):
    # Function detects faces and returns the cropped face
    # If no face detected, it returns the input image

    # gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(img, 1.3, 5)

    if faces is ():
        return None

    # Crop all faces found
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 2)
        cropped_face = img[y:y + h, x:x + w]

    return cropped_face


# Doing some Face Recognition with the webcam
video_capture = cv2.VideoCapture(0)
while True:
    _, frame = video_capture.read()
    # canvas = detect(gray, frame)
    # image, face =face_detector(frame)

    face = face_extractor(frame)
    if type(face) is np.ndarray:
        face = cv2.resize(face, (224, 224))
        im = PIL.Image.fromarray(face, 'RGB')
        # Resizing into 128x128 because we trained the model with this image size.
        img_array = np.array(im)
        # Our keras model used a 4D tensor, (images x height x width x channel)
        # So changing dimension 128x128x3 into 1x128x128x3
        img_array = np.expand_dims(img_array, axis=0)
        pred = model.predict(img_array)
        print(pred)
        name = "None matching"
        if (pred[0][0] > 0.5):
            name = 'Person 1'
        elif (pred[0][1] > 0.5):
            name= 'Person 2'
        elif (pred[0][2] > 0.5):
            name= 'Person 3'
        elif (pred[0][3] > 0.5):
            name= 'Person 4'
        cv2.putText(frame, name, (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
        # Track consecutive detections of the same face
        if detected_names and name == detected_names[-1]:
            consecutive_frames += 1
        else:
            consecutive_frames = 1
        # Append the detected name to the deque
        detected_names.append(name)
        # If the same face has been detected for the required number of frames
        if consecutive_frames >= max_consecutive_frames:
            # Write the detected name to the CSV file
            csv_writer.writerow([name])
            # Reset consecutive frames counter
            consecutive_frames = 0
    else:
        # Reset consecutive frames counter if no face is detected
        consecutive_frames = 0
        detected_names.clear()  # Clear deque
        # Display the frame with annotations
    cv2.imshow('Video', frame)
    # Exit loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# Close CSV file
csv_file.close()
    # else:
    #     cv2.putText(frame, "No face found", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
    # cv2.imshow('Video', frame)
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break
video_capture.release()
cv2.destroyAllWindows()
