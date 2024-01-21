import cv2

# HOGディテクタを初期化
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# 画像を読み込む
image = cv2.imread('path/to/your/image.jpg')

# 人物を検出
boxes, weights = hog.detectMultiScale(image, winStride=(8, 8), padding=(4, 4), scale=1.05)

# 検出された人物に四角形を描画
for (x, y, w, h) in boxes:
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

# 人物の数を表示
num_people = len(boxes)
print(f'検出された人物の数: {num_people}')

# 結果を表示
cv2.imshow('Person Detection', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
