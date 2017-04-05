"""This file contains the fonction to find frontal human faces in an image and estimate their pose.  The pose takes the form of 68 landmarks.  These are
points on the face such as the corners of the mouth, on the eyes, and so forth.

.. note::

 You can get the shape_predictor_68_face_landmarks.dat file from:
 http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2


"""
import dlib


class FaceLandmarks:

    # initialization
    def __init__(self, predictor_path, show_images):
        self.predictor_path = predictor_path
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(predictor_path)
        if show_images == True:
            self.win = dlib.image_window()
        else:
            self.win = None

    def detect(self, image):
        """ This function detect the face in the image.

        :param image: (image) Image being detected.
        :return:  features ({'points' : [], 'dists'  : []}):  The feature detected.

        """

        # self.image = numpy.copy(image)

        dets = self.detector(image)
        if len(dets)!=1:
            return None # warning more faces within the image
        else:
            shape = self.predictor(image, dets[0])
            # print dets[0].left(),dets[0].top(),dets[0].right(),dets[0].bottom()
            # for pt in shape.parts():
            #    self.features['points'].append((pt.x, pt.y))
            if self.win!=None:
                # Draw the face landmarks on the screen.
                self.win.clear_overlay()
                self.win.set_image(image)
                self.win.add_overlay(shape)
            features = {'points' : [], 'dists'  : []}
            for pt in shape.parts():
                features['points'].append((pt.x, pt.y))
            return features