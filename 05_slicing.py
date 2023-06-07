"""
5.  점자 영역 설정 findcontours함수를 활용하여 하려고 했지만 안됨
    고로 직접 그려주기로함 100단위로 선을 그음
"""
import cv2
import os
import numpy as np

img = cv2.imread('brailleimage/braille4_median.png', cv2.IMREAD_COLOR)
if img is None:
    raise Exception("ERROR")

output_folder = 'brailleimage/output_img'
os.makedirs(output_folder, exist_ok=True)

# 세로선 구분을 위한 좌표 계산
interval = 100  # 세로선 간격
separators = np.arange(interval, img.shape[1], interval)

# 세로선을 따라 이미지 슬라이스하여 저장
for i, sep in enumerate(separators):
    if i < len(separators) - 1:
        sub_img = img[:, separators[i]:separators[i+1]]
        cv2.imwrite(os.path.join(output_folder, f"output_image_{i+1}.jpg"), sub_img)


for i in range(100, img.shape[1], 100):
    cv2.line(img, (i, 0), (i, img.shape[0]), (0, 0, 0), 3)

cv2.imshow('Image with Lines', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
