from deepface import DeepFace
import matplotlib.pyplot as plt
from PIL import Image
import time
import cv2
import numpy as np

# Input image path
img_path = "./data/grp/group11.jpeg"

# Resize the image for better face detection
image = Image.open(img_path)
image = image.resize((640, 640))  # Resize to a manageable resolution
image.save("resized_image.jpg")
img_path = "resized_image.jpg"

detectors = ['yolov8']

for detector in detectors:
    print(f"Using detector: {detector}")
    tic = time.time()

    # Read image for drawing boxes
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    faces = DeepFace.extract_faces(img_path=img_path, detector_backend=detector, enforce_detection=False)
    toc = time.time()

    if len(faces) > 0:

        for i, face_data in enumerate(faces):
            face = face_data["face"]
            """Display individual detected faces
                        plt.figure(figsize=(5, 5))
                        plt.imshow(face)
                        plt.axis("off")
                        plt.title(f"{detector} Detected Face {i + 1}")
                        plt.show()
                        """

            # facial area coordinates
            facial_area = face_data["facial_area"]
            x = facial_area['x']
            y = facial_area['y']
            w = facial_area['w']
            h = facial_area['h']

           # bounding box with label
            cv2.rectangle(img_rgb, (x, y), (x + w, y + h), (0, 255, 0), 1)


            label = f"Face {i + 1}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img_rgb, label, (x, y - 10), font, 0.5,(0, 255, 0) , 1)



        # bounding boxes
        plt.figure(figsize=(10, 10))
        plt.imshow(img_rgb)
        plt.axis('off')
        plt.title(f' {len(faces)} Faces Detected using {detector}')
        plt.show()


    else:
        print(f"No faces detected with {detector} backend.")

    print(f"{detector} backend took {toc - tic:.2f} seconds and recognized {len(faces)} faces.")