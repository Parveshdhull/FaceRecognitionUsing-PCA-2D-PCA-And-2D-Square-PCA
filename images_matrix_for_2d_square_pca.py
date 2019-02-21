import cv2
import numpy as np


class images_to_matrix_class_for_two_d:


    def __init__(self, images_name, img_width, img_height):

        self.images_name = images_name
        self.img_width = img_width
        self.img_height = img_height
        self.img_size = (img_width * img_height)



    def get_matrix(self):

        img_mat = np.zeros(
            (len(self.images_name), self.img_height ,self.img_width),
            dtype=np.uint8)

        i = 0
        for name in self.images_name:
            # print("Name", name)
            gray = cv2.imread(name, 0)
            gray = cv2.resize(gray, (self.img_height, self.img_width))
            mat = np.asmatrix(gray)
            img_mat[i, :, :] = mat
            i += 1
        print("Matrix Size:", img_mat.shape)
        return img_mat
