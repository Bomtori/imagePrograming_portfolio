"""
2.  원근 투시 변환을 마친 이미지를 rotate 시켜줍니다.
"""
import cv2
from matplotlib import pyplot as plt

img = cv2.imread('brailleimage/braille1_transformed.png', cv2.IMREAD_GRAYSCALE)
if img is None: raise Exception("ERROR")

rotated = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
cv2.imwrite("brailleimage/braille2_rotated.png", rotated)

# 결과 출력
cv2.imshow('Rotated Image', rotated)
cv2.waitKey(0)
cv2.destroyAllWindows()
