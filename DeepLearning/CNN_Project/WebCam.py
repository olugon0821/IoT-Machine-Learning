import cv2
import numpy as np
import tensorflow as tf

# 모델 불러오기
model = tf.keras.models.load_model('TL.h5')

# OpenCV 비디오 캡처
cap = cv2.VideoCapture(0)  # 0은 기본 카메라를 의미합니다. 다른 카메라를 사용하려면 인덱스를 변경하세요.

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 프레임 크기를 모델 입력 크기로 조정
    resized_frame = cv2.resize(frame, (64, 64))
    # 모델 입력 형식으로 변환
    input_frame = np.expand_dims(resized_frame, axis=0)

    # 모델로 예측 수행
    predictions = model.predict(input_frame)
    # 가장 높은 확률의 클래스 선택
    predicted_class = np.argmax(predictions)

    # 예측 결과를 텍스트로 변환
    if predicted_class == 0:
        label = "Red"
        color = (0, 0, 255)  # 빨간색
    elif predicted_class == 1:
        label = "Green"
        color = (0, 255, 0)  # 초록색
    else:
        label = "Yellow"
        color = (0, 255, 255)  # 노란색

    # 예측 결과 텍스트를 프레임에 표시
    cv2.putText(frame, label, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    # 컨투어 검출
    gray = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 100, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 컨투어를 감싸는 경계 상자 그리기
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 1000:  # 일정 크기 이상의 컨투어만 사용
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

    # 비디오 출력
    cv2.imshow('Traffic Light Detection', frame)

    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 종료
cap.release()
cv2.destroyAllWindows()
