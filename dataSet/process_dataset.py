"""
This module is responsible for harvesting CK database for images of emotions. It gets a neutral face and a emotion face for each subject.
Based on Paul van Gent's code from blog post: http://www.paulvangent.com/2016/04/01/emotion-recognition-with-python-opencv-and-a-face-dataset/
"""
import glob
from shutil import copyfile

emotions = ["neutral", "anger", "contempt", "disgust", "fear", "happy", "sadness", "surprise"]  # Define emotion order
participants = glob.glob("source_emotion//*")  # Returns a list of all folders with participant numbers

for x in participants:
    part = "%s" % x[-4:]  # store current participant number
    for sessions in glob.glob("%s//*" % x):  # Store list of sessions for current participant
        for files in glob.glob("%s//*" % sessions):
            current_session = files[20:-30]
            file = open(files, 'r')

            emotion = int(
                float(file.readline()))  # emotions are encoded as a float, readline as float, then convert to integer.

            sourcefile_emotion = glob.glob("source_images//%s//%s//*" % (part, current_session))[
                -1]  # get path for last image in sequence, which contains the emotion
            sourcefile_emotion2 = glob.glob("source_images//%s//%s//*" % (part, current_session))[
                -2]  # get path for last image in sequence, which contains the emotion
            sourcefile_emotion3 = glob.glob("source_images//%s//%s//*" % (part, current_session))[
                -3]  # get path for last image in sequence, which contains the emotion
            sourcefile_emotion = glob.glob("source_images//%s//%s//*" % (part, current_session))[
                -4]  # get path for last image in sequence, which contains the emotion
            sourcefile_neutral = glob.glob("source_images//%s//%s//*" % (part, current_session))[
                0]  # do same for neutral image

            dest_neut = "sorted_set//neutral//%s" % sourcefile_neutral[25:]  # Generate path to put neutral image
            dest_emot = "sorted_set//%s//%s" % (
            emotions[emotion], sourcefile_emotion3[25:])  # Do same for emotion containing image

            #copyfile(sourcefile_neutral, dest_neut)  # Copy file
            copyfile(sourcefile_emotion3, dest_emot)  # Copy file
            #copyfile(sourcefile_emotion3, dest_emot)