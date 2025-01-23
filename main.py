import cv2
import os
from recogntion import FaceRecognizer
import matplotlib.pyplot as plt

dataset_path = "./data/dataset"
test_image_path = "./crazy/group/boys9.jpeg"

def main():
    recognizer = FaceRecognizer()
    recognizer.train(dataset_path)
    test_image = cv2.imread(test_image_path)
    if test_image is None:
        print(f"Error: Image not found at {test_image_path}")
        return
    faces, names = recognizer.recognize_faces(test_image)
    print(f"Detected {len(faces)} faces.")
    result_image = recognizer.draw_results(test_image, faces, names)
    plt.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title('Face Recognition Results')
    plt.show()

if __name__ == "__main__":
    main()