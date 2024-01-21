import cv2

# Haar Cascade分類器のパス
cascade_path = '../haarcascades/haarcascade_fullbody.xml'

# カスケード分類器を初期化
body_cascade = cv2.CascadeClassifier(cascade_path)

# USBカメラを初期化
cap = cv2.VideoCapture(1)  # カメラのデバイス番号（通常は0）に応じて調整

while True:
    # フレームを読み込む
    ret, frame = cap.read()
    if not ret:
        print("カメラからの映像を読み込めませんでした。")
        break

    # グレースケールに変換
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 人の体を検出
    bodies = body_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # 検出された人の体に四角形を描画
    for (x, y, w, h) in bodies:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # 人の体の数を表示
    num_bodies = len(bodies)
    cv2.putText(frame, f'検出された人の体の数: {num_bodies}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # 結果を表示
    cv2.imshow('Body Detection', frame)

    # 'q'を押すとループから抜けて終了
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# カメラを解放
cap.release()
cv2.destroyAllWindows()
