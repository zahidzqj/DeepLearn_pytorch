# _*_ coding: utf-8 _*_
#
import cv2
class bbx:
	def __init__(self, x1,y1,x2,y2,score):
		self.x1 = x1
		self.x2 = x2
		self.y1 = y1
		self.y2 = y2
		self.score = score

def iou1(a1,b1):
	max_x = max(a1.x1,b1.x1)
	max_y = max(a1.y1,b1.y1)
	min_x = min(a1.x2,b1.x2)
	min_y = min(a1.y2,b1.y2)
	s_i = (min_x-max_x)*(min_y-max_y)
	s_a = (a1.x2-a1.x1)*(a1.y2-a1.y1)
	s_b = (b1.x2-b1.x1)*(b1.y2-b1.y1)
	s_u = s_a+s_b-s_i
	return float(s_i/s_u)

def Coordinate(a1):
	fp = tuple([a1.x1,a1.y1])
	lp = tuple([a1.x2,a1.y2])
	return fp,lp

def nms1(bbx_list,k,threshold):
	bbx_list = sorted(bbx_list, key=lambda x:x.score, reverse = True)
	nms_list = [bbx_list[0]]
	print(len(bbx_list))
	print(len(nms_list))
	temp_lists=[]
	for i in range(k):
		for j in range(1,len(bbx_list)):
			iou = iou1(nms_list[i],bbx_list[j])
			if iou < threshold:
				temp_lists.append(bbx_list[j])
		if len(temp_lists) ==0:
			return nms_list
		bbx_list = temp_lists
		temp_lists = []
		nms_list.append(bbx_list[0])
	return nms_list
box1 = bbx(187, 82, 337, 317, 0.9)
box2 = bbx(150, 67, 305, 282, 0.75)
box3 = bbx(246, 121, 368, 304, 0.8)
bbx_list = [box1, box2, box3]
threshold = 0.4
nms_list =nms1(bbx_list,2,threshold)

image_name = 'nms.jpg'
image = cv2.imread(image_name)
org = image.copy()

for i in bbx_list:
	first_point,last_point=Coordinate(i)
	cv2.rectangle(org, first_point, last_point, (0, 255, 0), 2)
for x in nms_list:
	first_point,last_point=Coordinate(x)
	cv2.rectangle(image, first_point, last_point, (0, 255, 0), 2)

cv2.imshow('Original',org)
cv2.imshow('NMS_iou', image)
cv2.waitKey(0)



'''
def zuobiao(a1):
	fp = tuple(a1[0:2])
	lp = tuple(a1[2:])
	return fp,lp
a1=[100, 100, 200, 200] 
first_point,last_point=zuobiao(a1)

import cv2
import numpy as np

image_name = 'nms.jpg'
image = cv2.imread(image_name)

cv2.rectangle(image, first_point, last_point, (0, 255, 0), 2)#坐标需要时tuple


cv2.imshow('Original',image)
#cv2.imwrite(image_name, image)#保存处理的图片
cv2.waitKey(0)
'''

