"""This class is used to detect faces in a picture.

"""

from FaceLand import FaceLandmarks
import os
from skimage import io,data,color,img_as_ubyte

def extractFeatures(basepath_train, show_images):
    """This function detect landmarks for a dataset of faces.

    :param basepath_train: (str) Training data storage path.
    :param show_images: (bool) Whether to display the results after detecting the face.
    :return:         It return a tuple (res, facedb) where res is to check is all was ok (False if there are more than one face on the image) and facedb contains the images, together with coordinates points of landmarks for each image

    """

    res = True
    facedb = {}
    i = 1
    #initialization of the face landmarks
    fl = FaceLandmarks('shape_predictor_68_face_landmarks.dat', show_images)
    for x in os.listdir(basepath_train):
        pathn = basepath_train + '/' + x
        facedb[i] = []
        if os.path.isdir(pathn):
            for y in os.listdir(pathn):
                if y ==".DS_Store":
                    pass
                else:
                    # create detector object with a landmarks detector model
                    print (pathn + '/' + y)
                    img = io.imread(pathn + '/' + y)
                    # the second parameter is for drawing landmarks on the screen
                    img = color.rgb2gray(img)
                    img = img_as_ubyte(img)
                    features = fl.detect(img)
                    if features!=None:
                        facedb[i].append(features)
                    else:
                        res = False
        i = i + 1
    
    return (res, facedb)


# The function for IHMEmotionDetection
def getFeatures(basepath_train, show_images):
    """This function detect landmarks for a dataset of faces.

    :param basepath_train: (str) Training data storage path.
    :param show_images: (bool) Whether to display the results after detecting the face.
    :return:         It return a tuple (res, facedb) where res is to check is all was ok (False if there are more than one face on the image) and facedb contains the images, together with coordinates points of landmarks for each image

    """
    res = True
    facedb = {}
    facedb[1] = []
    # initialization of the face landmarks
    fl = FaceLandmarks('/Users/Alex/Documents/PRD/IHM_EmotionDetection/ProjetSI_FinalVersion/Ressources/Algorithmes/src/shape_predictor_68_face_landmarks.dat', show_images)
    img = io.imread(basepath_train)
    # the second parameter is for drawing landmarks on the screen
    features = fl.detect(img)
    if features != None:
        facedb[1].append(features)
    else:
        res = False
    return (res, facedb)