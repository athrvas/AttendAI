import insightface
from insightface.app import FaceAnalysis
import cv2


class FaceDetector:
    def __init__(self):
        self.detector = FaceAnalysis(providers=['CPUExecutionProvider'])
        self.detector.prepare(ctx_id=0, det_size=(640, 640))

    def detect_faces(self, image):
        return self.detector.get(image)

    def draw_faces(self, image, faces):
        img_draw = image.copy()
        for face in faces:
            bbox = face.bbox.astype(int)
            cv2.rectangle(img_draw, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
        return img_draw

    def extract_faces(self, image, faces):
        face_images = []
        for face in faces:
            bbox = face.bbox.astype(int)
            face_img = image[bbox[1]:bbox[3], bbox[0]:bbox[2]]
            face_images.append(face_img)
        return face_images
