import cv2
import time
import numpy as np

from PCA import pca_class
from images_to_matrix import images_to_matrix_class
from dataset import dataset_class


#for single image = 0
#for video = 1
#for group image = 2
reco_type = 0

#No of images For Training(Left will be used as testing Image)
no_of_images_of_one_person = 8
dataset_obj = dataset_class(no_of_images_of_one_person)


#Data For Training
images_names = dataset_obj.images_name_for_train
y = dataset_obj.y_for_train
no_of_elements = dataset_obj.no_of_elements_for_train
target_names = dataset_obj.target_name_as_array

#Data For Testing
images_names_for_test = dataset_obj.images_name_for_test
y_for_test = dataset_obj.y_for_test
no_of_elements_for_test = dataset_obj.no_of_elements_for_test


training_start_time = time.process_time()
img_width, img_height = 50, 50
i_t_m_c = images_to_matrix_class(images_names, img_width, img_height)
scaled_face = i_t_m_c.get_matrix()
cv2.imshow("Original Image" , cv2.resize(np.array(np.reshape(scaled_face[:,1],[img_height, img_width]), dtype = np.uint8),(200, 200)))
cv2.waitKey()

#PCA

my_pca = pca_class(scaled_face.T, y, target_names, no_of_elements, 90)
new_coordinates_pca = my_pca.reduce_dim()
my_pca.show_eigen_face(img_width, img_height, 50, 150, 0)

cv2.imshow("After PCA Image" , cv2.resize(np.array(np.reshape(my_pca.original_data(new_coordinates_pca[1, :]),[img_height, img_width]), dtype = np.uint8),(200, 200)))
cv2.waitKey()


training_time = time.process_time() - training_start_time


#Reco
if reco_type is 0:
    time_start = time.process_time()

    correct = 0
    wrong = 0
    i = 0
    net_time_of_reco = 0

    for img_path in images_names_for_test:

        time_start = time.process_time()
        find_name, find_y = my_pca.recognize_face(my_pca.new_cord(img_path, img_height, img_width))
        time_elapsed = (time.process_time() - time_start)
        net_time_of_reco += time_elapsed
        rec_y = y_for_test[i]
        rec_name = target_names[rec_y]
        if find_name is rec_name:
            correct += 1
            print("Correct", " Name:", find_name)
        else:
            wrong +=1
            print("Wrong:", " Real Name:", rec_name, "Rec Y:", rec_y, "Find Name:", find_name, "Find Y:", find_y)
        i+=1

    print("Correct", correct)
    print("Wrong", wrong)
    print("Total Test Images", i)
    print("Percent", correct/i*100)
    print("Total Person", len(target_names))
    print("Total Train Images", no_of_images_of_one_person * len(target_names))
    print("Total Time Taken for reco:", time_elapsed)
    print("Time Taken for one reco:", time_elapsed/i)
    print("Training Time", training_time)



#For Video

if reco_type is 1:
    face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=7)

        i = 0
        for(x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            scaled = cv2.resize(roi_gray, (img_height, img_width))
            rec_color = (255, 0, 0)
            rec_stroke = 2
            cv2.rectangle(frame, (x, y), (x+w, y+h), rec_color, rec_stroke)

            new_cord = my_pca.new_cord_for_image(scaled)
            name, min = my_pca.recognize_face(new_cord, i)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_color = (255, 255, 255)
            font_stroke = 2
            cv2.putText(frame, name + str(i), (x, y), font, 1, font_color, font_stroke, cv2.LINE_AA)
            i += 1



        cv2.imshow('Colored Frame', frame)
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break


    cap.release()
    cv2.destroyAllWindows()


#For Image

if reco_type is 2:
    face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
    dir = r'images/Group/'


    frame = cv2.imread(dir+ "group_image.jpg")


    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)


    i = 0

    for(x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        scaled = cv2.resize(roi_gray, (img_height, img_width))
        rec_color = (0, 255, 0)
        rec_stroke = 5
        cv2.rectangle(frame, (x, y), (x+w, y+h), rec_color, rec_stroke)

        new_cord = my_pca.new_cord_for_image(scaled)
        print("New Cord PCA"+str(i), new_cord)
        name, min = my_pca.recognize_face(new_cord, i)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_color = (255, 0, 0)
        font_stroke = 5
        cv2.putText(frame, name + str(i), (x, y), font, 8, font_color, font_stroke, cv2.LINE_AA)
        i += 1
        # cv2.imshow('Face', scaled)
        # cv2.waitKey()


    frame = cv2.resize(frame, (1080, 568))
    cv2.imshow('Colored Frame', frame)
    cv2.waitKey()






