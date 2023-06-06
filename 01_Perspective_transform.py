"""
1. 이미지를 받아온 후, 이미지의 원근 투시 과정을 거칩니다.
원근 변환을 거친 이미지를 저장합니다.
"""
import cv2
import numpy as np

#점자 이미지 불러오기
img = cv2.imread('brailleimage/braille1.png', cv2.IMREAD_GRAYSCALE)
if img is None: raise Exception("ERROR")

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#원근 투시(투영) 변환 Chapter8
def contain_pts(p, p1, p2):
    return p1[0] <= p[0] < p2[0] and p1[1] <= p[1] < p2[1]

#좌표 사각형 그리기 함수
def draw_rect(img):
    rois = [(p - small, small * 2) for p in pts1]
    for (x, y), (w, h) in np.int32(rois):
        roi = img[y : y + h, x : x + w]
        val = np.full(roi.shape, 80, np.uint8)
        cv2.add(roi, val, roi)
        cv2.rectangle(img, (x, y, w, h), (0, 255, 0), 1)
    cv2.polylines(img, [pts1.astype(int)], True, (0, 255, 0), 1)
    cv2.imshow("select rect", img)

# 마우스 드래그로 선택된 4개 좌표에 목적 영상 4개 좌표로 원근 변환 행렬 계산
def draw_rect(img):
    rois = [(p - small, small * 2) for p in pts1]
    for (x, y), (w, h) in np.int32(rois):
        roi = img[y : y + h, x : x + w]
        val = np.full(roi.shape, 80, np.uint8)
        cv2.add(roi, val, roi)
        cv2.rectangle(img, (x, y, w, h), (0, 255, 0), 1)
    cv2.polylines(img, [pts1.astype(int)], True, (0, 255, 0), 1)
    cv2.imshow("select rect", img)

def warp(img):
    perspect_mat = cv2.getPerspectiveTransform(pts1, pts2)
    dst = cv2.warpPerspective(img, perspect_mat, (200, 800), cv2.INTER_CUBIC)
    cv2.imshow("perspective transform", dst)
    cv2.imwrite("brailleimage/braille1_transformed.png", dst)  # 영역 처리된 파일 저장


def onMouse(event, x, y, flags, param):
    global check
    if event == cv2.EVENT_LBUTTONDOWN:
        for i, p in enumerate(pts1):
            p1, p2 = p - small, p + small
            if contain_pts((x, y), p1, p2): check = i

    if event == cv2.EVENT_LBUTTONUP: check = -1

    if check >= 0:
        pts1[check] = (x, y)
        draw_rect(np.copy(img))
        warp(np.copy(img))


small = np.array([12, 12])
check = -1
pts1 = np.float32([(10, 10), (10, 20), (20, 10), (20, 20)])
pts2 = np.float32([(0, 0), (200, 0), (200, 800), (0, 800)])

draw_rect(np.copy(img))
cv2.setMouseCallback("select rect", onMouse, 0)
cv2.waitKey(0)


