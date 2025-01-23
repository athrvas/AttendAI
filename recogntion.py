import os
import numpy as np
import cv2
from detection import FaceDetector


class FaceRecognizer:
    def __init__(self):
        self.detector = FaceDetector()
        self.known_embeddings = []
        self.known_names = []

    def train(self, dataset_path):
        print("Training model with dataset...")
        for person_name in os.listdir(dataset_path):
            person_dir = os.path.join(dataset_path, person_name)
            if os.path.isdir(person_dir):
                for img_name in os.listdir(person_dir):
                    img_path = os.path.join(person_dir, img_name)
                    try:
                        img = cv2.imread(img_path)
                        faces = self.detector.detect_faces(img)
                        if len(faces) > 0:
                            self.known_embeddings.append(faces[0].embedding)
                            self.known_names.append(person_name)
                    except Exception as e:
                        print(f"Error processing {img_path}: {str(e)}")
        self.known_embeddings = np.array(self.known_embeddings)
        print(f"Training completed! Learned {len(self.known_names)} faces.")

    def recognize_faces(self, image, threshold=0.97):
        faces = self.detector.detect_faces(image)
        names = []
        for face in faces:
            embedding = face.embedding
            similarities = np.dot(self.known_embeddings, embedding)
            most_similar_idx = np.argmax(similarities)
            name = self.known_names[most_similar_idx] if similarities[most_similar_idx] > threshold else "Unknown"
            names.append(name)
        return faces, names

    def draw_results(self, image, faces, names):
        img_draw = image.copy()
        for face, name in zip(faces, names):
            bbox = face.bbox.astype(int)
            cv2.rectangle(img_draw, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
            cv2.putText(img_draw, name, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        return img_draw