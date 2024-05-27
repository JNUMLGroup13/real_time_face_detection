# spell-checker: disable

import dlib
import cv2
import os
import sys
import random
import numpy as np

outputDir = './myFace/paddy'
if not os.path.exists(outputDir):
    os.makedirs(outputDir)

# change image lightness and contrast
def reLight(img, alpha, beta):
    new_img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    new_img = np.clip(new_img, 0, 255)
    return new_img

detector = dlib.get_frontal_face_detector()
camera = cv2.VideoCapture(0)

index = 1
while True:
    if (index<= 2000):
        print('It\'s processing the ' + str(index) + 'th picture')
        success, img = camera.read()
        # change image lightness and contrast
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        dets = detector(gray_img, 1)

        for i, d in enumerate(dets):
            x1 = d.top() if d.top() > 0 else 0
            y1 = d.bottom() if d.bottom() > 0 else 0
            x2 = d.left() if d.left() > 0 else 0
            y2 = d.right() if d.right() > 0 else 0

            face = img[x1:y1, x2:y2]
            face = reLight(face, random.uniform(0.5, 1.5), random.randint(-50, 100))

            face = cv2.resize(face, (64, 64))

            cv2.imshow('image', face)
            cv2.imwrite(outputDir + '/' + str(index) + '.jpg', face)
            index += 1

        key = cv2.waitKey(30) & 0xff
        if key == 27:
            break
    else:
        print('Finished!')
        break

    