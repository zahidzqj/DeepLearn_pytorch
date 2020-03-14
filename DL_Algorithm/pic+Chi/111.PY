# -*- coding: utf-8 -*-   
import cv2
from   cv2dr  import *

img = cv2.imread('IMG123.jpg')
line = '你好'
#img = np.zeros([300,300,3])

color_ = (0,255,0) # Green
pos = (3, 3)
text_size = 24
#ft = put_chinese_text('wqy-zenhei.ttc')
ft = put_chinese_text('C:/Windows/Fonts/deng.ttf')
image = ft.draw_text(img, pos, line, text_size, color_)

cv2.imshow('ss', image)
cv2.waitKey(0)