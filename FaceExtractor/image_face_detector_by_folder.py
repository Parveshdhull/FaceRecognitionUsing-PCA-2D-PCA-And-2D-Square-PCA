import cv2
import numpy as np
import os
from os.path import dirname, abspath


class face_extractor:

    def __init__(self):

        d = dirname(dirname(abspath(__file__)))
        self.face_cascade1 = cv2.CascadeClassifier(d + '/cascades/data/haarcascade_frontalface_alt2.xml')
        self.face_cascade2 = cv2.CascadeClassifier(d + '/cascades/data/haarcascade_frontalface_alt.xml')
        self.face_cascade3 = cv2.CascadeClassifier(d + '/cascades/data/haarcascade_profileface.xml')
        self.min_width = 2000
        self.min_height = 2000

        self.dir = d + r'/images/'  # Set Images Folders Location, It will find images in all folders
        print("Location:", self.dir)
        self.faces = []
        self.names = []


    def find_faces(self):

        for name in os.listdir(self.dir):
            temp_dir = os.path.join(self.dir, name)
            if os.path.isdir(temp_dir):
                if name in ["Cropped", "Originals", "Group Photos"]:    # Folders to execulde
                    continue
                for file_name in os.listdir(temp_dir):
                    temp = temp_dir + "/" +file_name
                    temp2 = temp_dir  + "/" + "_cropped" + file_name
                    img = cv2.imread(temp)
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    faces = self.face_cascade1.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=3)
                    if len(faces) == 0:
                        faces = self.face_cascade2.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=3)
                    if len(faces) == 0:
                        faces = self.face_cascade3.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=3)
                    if len(faces) == 0:
                        continue
                    i = 0
                    for face in faces:
                        x, y, w, h = face
                        cropped = gray[y: y + h, x: x + w]
                        if i == 0:
                            self.names += [temp2]
                            i += 1
                        else:
                            self.names += [temp2 + str(i) + ".png"]
                            i += 1
                        self.faces += [cropped]
                        if self.min_width > w:
                            self.min_width = w
                        if self.min_height > h:
                            self.min_height = h

            i = 0
            for face in self.faces:
                print("Face", self.names[i])
                resized_image = cv2.resize(face, (228, 228))
                cv2.imwrite(self.names[i], resized_image)
                i += 1

            self.faces = []
            self.names = []