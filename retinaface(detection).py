from retinaface import RetinaFace
import cv2
import matplotlib.pyplot as plt

img_path="./crazy/group/boys9.jpeg"
img=cv2.imread(img_path)
obj=RetinaFace.detect_faces(img_path)

for key in obj.keys():
    identity=obj[key]

    facial_area=identity['facial_area']
    cv2.rectangle(img,(facial_area[2], facial_area[3]),(facial_area[0],facial_area[1]),(255, 255, 255),2)

print(len(obj.keys()))
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()