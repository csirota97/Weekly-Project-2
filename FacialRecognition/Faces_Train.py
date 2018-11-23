import os
import cv2
import pickle
import numpy as np
from PIL import Image

face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
img_dir = os.path.join(BASE_DIR, "images")

recognizer = cv2.face.LBPHFaceRecognizer_create()

cur_id = 0
label_ids = {}
x_train = []
y_labels = []

for root, dirs, files in os.walk(img_dir):
    for file in files:
        if file.endswith("jpg") or file.endswith("png") or file.endswith("jpeg"):
            path = os.path.join(root, file)
            label = os.path.basename(root).replace(" ", "-").lower()
#            print(label, path)
            if not label in label_ids:
                label_ids[label] = cur_id
                cur_id += 1

            id_ = label_ids[label]
#            print(label_ids)

        #    x_train.append(path)  # verify this image, turn into a NUMPY array, gray
        #    y_labels.append(label)  # some numbers

            pil_image = Image.open(path).convert("L")  # grayscale
            size = (550, 550)
            final_img = pil_image.resize(size, Image.ANTIALIAS)
            img_array = np.array(final_img, "uint8")
#            print(img_array)
            faces = face_cascade.detectMultiScale(img_array, scaleFactor=1.5, minNeighbors=5)

            for (x, y, w, h) in faces:
                roi = img_array[y:y+h, x:x+w]
                x_train.append(roi)
                y_labels.append(id_)



# print(y_labels)
# print(x_train)

with open("labels.pickle", 'wb') as f:
    pickle.dump(label_ids, f)

recognizer.train(x_train, np.array(y_labels))
recognizer.save("trainner.yml")

