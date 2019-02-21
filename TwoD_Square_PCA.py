import numpy as np
import cv2
import scipy.linalg as s_linalg


class two_d_square_pca_class:


    def give_p(self, d):
        print("D", d)
        sum = np.sum(d)
        sum_85 = 0.95 * sum
        temp = 0
        p = 0
        while temp < sum_85:
            temp += d[p]
            p += 1
        return p

    def reduce_dim(self):

        no_of_images = self.images.shape[0]
        mat_height = self.images.shape[1]
        mat_width = self.images.shape[2]
        g_t = np.zeros((mat_height, mat_height))
        h_t = np.zeros((mat_width, mat_width))

        for i in range(no_of_images):
            temp = np.dot(self.images_mean_subtracted[i].T, self.images_mean_subtracted[i])
            g_t += temp
            h_t += np.dot(self.images_mean_subtracted[i], self.images_mean_subtracted[i].T)

        g_t /= no_of_images
        h_t /= no_of_images

        #For G_T
        d_mat, p_mat = np.linalg.eig(g_t)
        p_1 = self.give_p(d_mat)
        self.new_bases_gt = p_mat[:, 0:p_1]

        #For H_T
        d_mat, p_mat = np.linalg.eig(h_t)
        p_2 = self.give_p(d_mat)
        self.new_bases_ht = p_mat[:, 0:p_2]


        new_coordinates_temp = np.dot(self.images, self.new_bases_gt)

        self.new_coordinates = np.zeros((no_of_images, p_2, p_1))

        for i in range(no_of_images):
            self.new_coordinates[i, :, :] = np.dot(self.new_bases_ht.T, new_coordinates_temp[i])

        return self.new_coordinates


    def __init__(self, images, y, target_names):
        self.images = np.asarray(images)
        self.y = y
        self.target_names = target_names
        self.mean_face = np.mean(self.images, 0)
        self.images_mean_subtracted = self.images - self.mean_face


    def original_data(self, new_coordinates):
        return np.dot(self.new_bases_ht, np.dot(new_coordinates, self.new_bases_gt.T))

    def new_cord(self, name, img_height, img_width):
        img = cv2.imread(name, 0)
        cv2.imshow("Recognize Image",img)
        cv2.waitKey()
        gray = cv2.resize(img, (img_height, img_width))
        return np.dot(self.new_bases_ht.T, np.dot(gray, self.new_bases_gt))

    def new_cord_for_image(self, image):
        return np.dot(self.new_bases_ht.T, np.dot(gray, self.new_bases_gt))




    def recognize_face(self, new_cord):

        no_of_images = len(self.y)
        distances = []
        for i in range(no_of_images):
            temp_imgs = self.new_coordinates[i]
            dist = np.linalg.norm(new_cord - temp_imgs)
            distances += [dist]

        min = np.argmin(distances)
        per = self.y[min]
        per_name = self.target_names[per]
        if distances[min] < 14975:
            print("Person", per, ":", min, self.target_names[per], "Dist:", distances[min])
            return per_name
        else:
            print("Person", per, ":", min, 'Unknown', "Dist:", distances[min])
            return 'Un'






