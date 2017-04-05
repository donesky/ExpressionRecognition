#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''This module is responsible for adjusting the face in the image by the location of 2 eyes.
Copyright (c) Philipp Wagner. All rights reserved.
'''

import sys, math, Image
import cv2



def Distance(p1, p2):
    '''
    This function calculate the distance between 2 points.
    :param p1: point 1
    :param p2: point 2
    :return: distance
    '''
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    return math.sqrt(dx * dx + dy * dy)


def ScaleRotateTranslate(image, angle, center=None, new_center=None, scale=None, resample=Image.BICUBIC):
    '''
    This function adjusts the angle of the face.
    :param image: Image to be processed
    :param angle: The inclination angle of the face
    :param center: The center of the angle transformation
    :param new_center: New center
    :param scale: Scale factor
    :param resample: Image.BICUBIC
    :return: Transformed image
    '''
    if (scale is None) and (center is None):
        return image.rotate(angle=angle, resample=resample)
    nx, ny = x, y = center
    sx = sy = 1.0
    if new_center:
        (nx, ny) = new_center
    if scale:
        (sx, sy) = (scale, scale)
    cosine = math.cos(angle)
    sine = math.sin(angle)
    a = cosine / sx
    b = sine / sx
    c = x - nx * a - ny * b
    d = -sine / sy
    e = cosine / sy
    f = y - nx * d - ny * e
    return image.transform(image.size, Image.AFFINE, (a, b, c, d, e, f), resample=resample)


def CropFace(image, eye_left=(0, 0), eye_right=(0, 0), offset_pct=(0.2, 0.2), dest_sz=(70, 70)):
    '''

    :param image: Original image
    :param eye_left: Position of the left eye
    :param eye_right: Position of the right eye
    :param offset_pct: used for calculating offsets in original image
    :param dest_sz: size of cropped image
    :return: Cropped image
    '''
    # calculate offsets in original image
    offset_h = math.floor(float(offset_pct[0]) * dest_sz[0])
    offset_v = math.floor(float(offset_pct[1]) * dest_sz[1])
    # get the direction
    eye_direction = (eye_right[0] - eye_left[0], eye_right[1] - eye_left[1])
    # calc rotation angle in radians
    rotation = -math.atan(float(eye_direction[1]) / float(eye_direction[0]))
    # distance between them
    dist = Distance(eye_left, eye_right)
    # calculate the reference eye-width
    reference = dest_sz[0] - 2.0 * offset_h
    # scale factor
    scale = float(dist) / float(reference)
    # rotate original around the left eye
    image = ScaleRotateTranslate(image, center=eye_left, angle=rotation)
    # crop the rotated image
    crop_xy = (eye_left[0] - scale * offset_h, eye_left[1] - scale * offset_v)
    crop_size = (dest_sz[0] * scale, dest_sz[1] * scale)
    image = image.crop(
        (int(crop_xy[0]), int(crop_xy[1]), int(crop_xy[0] + crop_size[0]), int(crop_xy[1] + crop_size[1])))
    # resize it
    image = image.resize(dest_sz, Image.ANTIALIAS)
    return image

if __name__ == '__main__':
    image = Image.open("../IHM_Image/adjustImage.png")
    ic = ScaleRotateTranslate(image,30)
    #ic = ScaleRotateTranslate(image,90)
    #ic = ScaleRotateTranslate(image,-30)

    #ic=CropFace(image,(0,1))
    ic.show()