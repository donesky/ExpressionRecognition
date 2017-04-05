import cv2
import numpy as np
import emotionCamera as ec
from skimage import io,data,color,img_as_ubyte

def main():
    cap = cv2.VideoCapture(0)
    while(cap.isOpened()):
        ret, img       = cap.read()
        a = ec.expected_return_1(img)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, a, (100,200), font, 2, (255, 255, 255), 2)
        cv2.imshow('capture', img)

        k = cv2.waitKey(10)
        if k == 27:
            break

if __name__ == '__main__':
    main()