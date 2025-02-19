import cv2
import time
# === Загрузка моделей ===
# Viola-Jones (Haar Cascade)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
# SSD (Caffe модель для обнаружения лиц)
net = cv2.dnn.readNetFromCaffe(
    "C:/Users/User/PycharmProjects/pythonProject2/Vio/deploy.prototxt",
    "C:/Users/User/PycharmProjects/pythonProject2/Vio/res10_300x300_ssd_iter_140000.caffemodel"
)
# === Функция для тестирования Viola-Jones ===
def test_viola_jones(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    start = time.time()
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    end = time.time()
    return faces, end - start
# === Функция для тестирования SSD ===
def test_ssd(image):
    h, w = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    start = time.time()
    detections = net.forward()
    end = time.time()
    faces = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:  # Фильтруем по порогу уверенности
            box = detections[0, 0, i, 3:7] * [w, h, w, h]
            faces.append(box.astype("int"))
    return faces, end - start
# === Загрузка изображения ===
image_path = "C:/Users/User/Desktop/cefca2768c6581ecb4950296f73bdad5.jpg"
image = cv2.imread(image_path)

if image is None:
    print("Mistake with import image!")
else:
    # Тестирование Viola-Jones
    vj_faces, vj_time = test_viola_jones(image)
    # Тестирование SSD
    ssd_faces, ssd_time = test_ssd(image)

    print(f"Viola-Jones: {len(vj_faces)} faces, Time: {vj_time:.4f} сек")
    print(f"SSD: {len(ssd_faces)} faces, Time: {ssd_time:.4f} сек")

    # === Визуализация результатов ===
    for (x, y, w, h) in vj_faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Синие рамки для Viola-Jones

    for (x1, y1, x2, y2) in ssd_faces:
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Зеленые рамки для SSD

    # === Отображение изображения ===
    cv2.imshow("Face Detection Comparison", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
