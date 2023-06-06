"""
06. 슬라이싱한거 CNN 돌려서 학습한거 넣어서 정보 빼기
"""

import numpy as np
from pathlib import Path
import cv2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# 데이터셋 경로 설정
img_dir = Path('brailleimage/Braille Dataset')
output_dir = Path('brailleimage/output_img')

# 데이터셋 이미지 파일 경로 리스트 생성
braille_files = list(img_dir.glob('*.jpg'))
output_files = list(output_dir.glob('*.jpg'))
all_files = braille_files + output_files

# 이미지 데이터와 라벨 저장할 리스트 초기화
images = []
labels = []

# 이미지 데이터와 라벨 수집
for file in all_files:
    image = cv2.imread(str(file), cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (28, 28))  # 이미지 크기 조정
    images.append(image)

    # 파일 이름으로부터 영어 라벨 추출
    label = file.stem
    labels.append(label)

# 이미지 데이터와 라벨을 NumPy 배열로 변환
images = np.array(images) / 255.0  # 이미지를 0~1 사이의 값으로 정규화
labels = np.array(labels)

# 라벨 인코딩 및 One-Hot 인코딩
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)
labels_onehot = to_categorical(labels_encoded)

# 훈련 데이터와 테스트 데이터 분할
train_images, test_images, train_labels, test_labels = train_test_split(images, labels_onehot, test_size=0.2,
                                                                        random_state=42)

# CNN 모델 생성
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(26, activation='softmax'))

# 모델 컴파일 및 학습
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(train_images, train_labels, batch_size=128, epochs=10, validation_data=(test_images, test_labels))

# output_img의 점자 데이터 예측
output_images = []
for file in output_files:
    image = cv2.imread(str(file), cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (28, 28))
    output_images.append(image)
output_images = np.array(output_images) / 255.0

# 예측 수행
predictions = model.predict(output_images)
predicted_labels = label_encoder.inverse_transform(np.argmax(predictions, axis=1))

# 결과 출력
for i, file in enumerate(output_files):
    print(f"Image: {file.name}, Predicted Label: {predicted_labels[i]}")
