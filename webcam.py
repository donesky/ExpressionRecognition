"""
This module loads emotion recognition model from a file,
shows a webcam image, recognizes face and it's emotion and draw emotion on the image.
"""
from cv2 import WINDOW_NORMAL

import cv2




def show_webcam_and_run( window_size=None, window_name='webcam', update_time=10):
    """
    Shows webcam image, detects faces and its emotions in real time and draw emoticons over those faces.
    :param window_size: Size of webcam image window.
    :param window_name: Name of webcam image window.
    :param update_time: Image update time interval.
    """
    cv2.namedWindow(window_name, WINDOW_NORMAL)
    if window_size:
        width, height = window_size
        cv2.resizeWindow(window_name, width, height)

    vc = cv2.VideoCapture(0)
    if vc.isOpened():
        read_value, webcam_image = vc.read()
    else:
        print("webcam not found")
        return

    while read_value:

        cv2.imshow(window_name, webcam_image)
        read_value, webcam_image = vc.read()
        key = cv2.waitKey(update_time)

        if key == 27:  # exit on ESC
            break

    cv2.destroyWindow(window_name)


if __name__ == '__main__':
    #emotions = ['neutral', 'anger', 'disgust', 'happy', 'sadness', 'surprise']
    #load_emoticons
    # load model

    # use learnt model
    window_name = 'WEBCAM (press ESC to exit)'
    show_webcam_and_run( window_size=(1600, 1200), window_name=window_name, update_time=8)
