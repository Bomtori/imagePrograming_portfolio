"""
4.  더욱 깔끔한 이미지를 얻기 위해 잡음을 제거 후,
    소금, 후추 잡음을 잡기 위해 미디언 필터를 활용합니다.
"""
import cv2

img = cv2.imread('brailleimage/braille3_threshold.png', cv2.IMREAD_COLOR)
if img is None: raise Exception("ERROR")

img_median = cv2.medianBlur(img, 3)
cv2.imwrite('brailleimage/braille4_median.png', img_median)

cv2.imshow('img', img)
cv2.imshow('img_median', img_median)
cv2.waitKey(0)