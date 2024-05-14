import cv2
import numpy as np
import tensorflow as tf
import socket

HOST = '192.168.0.8'
PORT = 9999
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect((HOST, PORT))

def send_to_java_ui(data):
    #데이터 전송
    try:
        s.send(data.encode('UTF-8'))
    except Exception as e:
        print("Error while sending data to Java UI:", e)

def main():
    # 모델 불러오기
    model = tf.keras.models.load_model('TL5.h5')
    # OpenCV 비디오 캡처
    cap = cv2.VideoCapture(0)

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
            label = "red\n"
        elif predicted_class == 1:
            label = "green\n"
        else:
            label = "yellow\n"
        print(label)

        # 데이터 전송
        send_to_java_ui(label)

        # 비디오 출력
        cv2.imshow('Traffic Light Detection', frame)

        # 'q' 키를 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    s.close()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()