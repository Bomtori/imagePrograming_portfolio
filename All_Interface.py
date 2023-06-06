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

"""
3.  이미지를 흑백 이진 이미지로 변환시킵니다.
    적절한 임계값을 설정하기 위해 히스토그램을 확인 후 구분합니다.
"""

img = cv2.imread('brailleimage/braille2_rotated.png', cv2.IMREAD_COLOR)
if img is None: raise Exception("ERROR")

#CH06 히스토그램 그래프 그리기 예제 참고
hist = cv2.calcHist([img], [0], None, [256], [0, 256])

plt.plot(hist)
plt.show()

"""
히스토그램을 확인한 결과 150과 250사이에 값이 모여있는 것을 확인할 수 있다.
영상 이진화를 하기위해 trackbar와 threshold를 활용하여 영상을 적절하게 이진화를 진행합니다.
"""

def on_threshold(pos):
    _, dst = cv2.threshold(img, pos, 255, cv2.THRESH_BINARY)
    cv2.imshow('braille3_Threshold', dst)
    cv2.imwrite('brailleimage/braille3_threshold.png', dst)

#이진화 전 이미지를 보여줍니다.
cv2.imshow('braille2', img)
cv2.namedWindow('braille3_Threshold')

#CH04 트랙바 이벤트 제어 예제 참고
#트랙바의 임계값을 0 ~ 255로 지정합니다.
cv2.createTrackbar('Threshold', 'braille3_Threshold', 0, 255, on_threshold)
#트랙바 위치를 150로 반환합니다.
cv2.setTrackbarPos('Threshold', 'braille3_Threshold', 150)

cv2.waitKey(0)

"""
4.  더욱 깔끔한 이미지를 얻기 위해 잡음을 제거 후,
    소금, 후추 잡음을 잡기 위해 미디언 필터를 활용합니다.
"""

img = cv2.imread('brailleimage/braille3_threshold.png', cv2.IMREAD_COLOR)
if img is None: raise Exception("ERROR")

img_median = cv2.medianBlur(img, 3)
cv2.imwrite('brailleimage/braille4_median.png', img_median)

cv2.imshow('img', img)
cv2.imshow('img_median', img_median)
cv2.waitKey(0)

"""
5.  점자 영역 설정
"""

img = cv2.imread('brailleimage/braille1_median.png', cv2.IMREAD_COLOR)
if img is None: raise Exception("ERROR")



