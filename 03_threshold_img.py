"""
3.  이미지를 흑백 이진 이미지로 변환시킵니다.
    적절한 임계값을 설정하기 위해 히스토그램을 확인 후 구분합니다.
"""
import cv2
from matplotlib import pyplot as plt

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
