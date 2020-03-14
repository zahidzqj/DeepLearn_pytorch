# _*_ coding: utf-8 _*_

import cv2
import numpy as np


image_name = 'nms.jpg'

image = cv2.imread(image_name)

cro=128*np.ones((500,449,3),dtype='uint8')
cro[40:400,40:400,:]=image[40:400,40:400,:]
cro=cro-128

cv2.imshow('process', cro)
cv2.imshow('original', image)
cv2.waitKey(0)
