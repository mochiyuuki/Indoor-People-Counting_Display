import cv2
import numpy as np

# HOGディテクタを初期化
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# USBカメラを初期化
cap = cv2.VideoCapture(1)  # カメラのデバイス番号（通常は0）に応じて調整

while True:
    # フレームを読み込む
    ret, frame = cap.read()
    if not ret:
        print("カメラからの映像を読み込めませんでした。")
        break

    # 人を検出
    bodies, weights = hog.detectMultiScale(frame, winStride=(8, 8), padding=(4, 4), scale=1.05)

    # 検出された人に四角形を描画
    for (x, y, w, h) in bodies:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # 人の数を表示
    num_people = len(bodies)
    cv2.putText(frame, f'検出された人の数: {num_people}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # 結果を表示
    cv2.imshow('Person Detection', frame)

    # 'q'を押すとループから抜けて終了
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# カメラを解放
cap.release()
cv2.destroyAllWindows()
