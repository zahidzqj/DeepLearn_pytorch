# _*_ coding: utf-8 _*_


import cv2
import numpy as np


# Image name
image_name = 'nms.jpg'

# Bounding boxes
bounding_boxes = [(187, 82, 337, 317), (150, 67, 305, 282), (246, 121, 368, 304)]
confidence_score = [0.9, 0.75, 0.8]


# Read image
image = cv2.imread(image_name)

# Copy image as original
org = image.copy()


first_point = (100, 100)
last_point = (200, 200)
baseline=0
w=0
#for (start_x, start_y, end_x, end_y) in bounding_boxes:
#for (start_x, start_y, end_x, end_y), confidence in zip(bounding_boxes, confidence_score):
	#cv2.rectangle(image,(start_x, start_y - (2 * baseline + 5)), (start_x + w, start_y), (0, 255, 255), -1)
	#cv2.rectangle(image, (start_x, start_y), (end_x, end_y), (0, 255, 255), 2)

cv2.rectangle(image, first_point, last_point, (0, 255, 0), 2)

#cv2.imwrite(image_name, image)#保存处理的图片
cv2.imshow('Original',image)
cv2.waitKey(0)
'''
# Draw parameters
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1
thickness = 2


# Show image
cv2.imshow('Original', org)
cv2.imshow('NMS', image)
cv2.waitKey(0)
'''

