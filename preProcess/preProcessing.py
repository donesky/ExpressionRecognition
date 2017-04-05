'''This module is responsible for preprocessing of images. The purpose is to improve the recognition rate of expression.
'''

import cv2
import dlib
import numpy
import sys, math, Image
import adjust
import os
from skimage import io

PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
basepath_train="ImageData/ImagesTest"
detector = dlib.get_frontal_face_detector()

predictor = dlib.shape_predictor(PREDICTOR_PATH)
#left eye location
eye_left_x = 0
eye_left_y = 0
#right eye location
eye_right_x = 0
eye_right_y = 0

class NoFaces(Exception):
    pass

# initialization of the face landmarks
for x in os.listdir(basepath_train):
    pathn = basepath_train + '/' + x
    if os.path.isdir(pathn):
        for y in os.listdir(pathn):
            # create detector object with a landmarks detector model
            print (pathn + '/' + y)
            im = cv2.imread(pathn + '/' + y)
            rects = detector(im, 1)

            if len(rects) >= 1:
                print("{} faces detected".format(len(rects)))

            if len(rects) == 0:
                raise NoFaces

            for i in range(len(rects)):
                landmarks = numpy.matrix([[p.x, p.y] for p in predictor(im, rects[i]).parts()])
                im = im.copy()
                index = 0
                for idx, point in enumerate(landmarks):
                    # index of left eye
                    if index == 36:
                        eye_left_x = point[0, 0]
                        eye_left_y = point[0, 1]
                    # index of right eye
                    if index == 45:
                        eye_right_x = point[0, 0]
                        eye_right_y = point[0, 1]
                    index += 1
                    #pos = (point[0, 0], point[0, 1])
                    #cv2.circle(im, pos, 3, color=(0, 255, 0))

            #cv2.namedWindow("im", 2)
            #cv2.imshow("im", im)
            #cv2.waitKey(0)
            image = Image.open(pathn + '/' + y)
            adjust.CropFace(image, eye_left=(eye_left_x, eye_left_y), eye_right=(eye_right_x, eye_right_y),offset_pct=(0.2, 0.2), dest_sz=(200, 200)).save('ImageData/AdjustAllImage/'+x+'/'+y)








