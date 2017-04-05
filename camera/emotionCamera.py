import sys
import getLatentSpace
import matplotlib;matplotlib.rcParams['figure.figsize'] = (8,6)
import matplotlib.pyplot as plt
import dataSet.setFeaturesData as sfd
import classify.knn as knn
import cv2
import dlib
import numpy
import sys, math, Image
import preProcess.adjust as adjust


PREDICTOR_PATH = "/Users/Alex/Documents/PRD/IHM_EmotionDetection/ProjetSI_FinalVersion/Ressources/Algorithmes/src/shape_predictor_68_face_landmarks.dat"


def expected_return_1(image):
    detector = dlib.get_frontal_face_detector()

    predictor = dlib.shape_predictor(PREDICTOR_PATH)
    # left eye location
    eye_left_x = 0
    eye_left_y = 0
    # right eye location
    eye_right_x = 0
    eye_right_y = 0

    class NoFaces(Exception):
        pass

    # initialization of the face landmarks

    im = image
    rects = detector(im, 1)
    if len(rects) == 0:
        return 'No face'
    else:
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
                # pos = (point[0, 0], point[0, 1])
                # cv2.circle(im, pos, 3, color=(0, 255, 0))

                # cv2.namedWindow("im", 2)
                # cv2.imshow("im", im)
                # cv2.waitKey(0)
        image = Image.fromarray(im)
        adjust.CropFace(image, eye_left=(eye_left_x, eye_left_y), eye_right=(eye_right_x, eye_right_y),
                        offset_pct=(0.2, 0.2), dest_sz=(200, 200)).save('/Users/Alex/Documents/PRD/IHM_EmotionDetection/ProjetSI_FinalVersion/Ressources/Algorithmes/src/IHM_Image/adjustImage.png')

        # get data from test images
        # dataAsListTest, labelsAsListTest = sfd.getListsFromImages('ImagesTest')
        dataAsListTest, labelsAsListTest = sfd.getListFromImage('/Users/Alex/Documents/PRD/IHM_EmotionDetection/ProjetSI_FinalVersion/Ressources/Algorithmes/src/IHM_Image/adjustImage.png')

        # create space latent by test data and train data
        LatentModel, dataSamples, labelSamples = getLatentSpace.genLatentSpaceTest(dataAsListTest, labelsAsListTest, 100, 2)
        # LatentModel, dataSamples, labelSamples = getLatentSpace.genLatentSpace(100, 2)

        mu, var = LatentModel.predict(dataSamples)
        numPoint = LatentModel.num_data
        listPoint = [map(float, result[0:2]) for result in LatentModel._predictive_variable[0:numPoint]]
        # get the predict Matrix using method KNN
        outlabel=knn.predictImage(listPoint, labelSamples, len(labelsAsListTest), 4, labelsAsListTest)
        return outlabel


