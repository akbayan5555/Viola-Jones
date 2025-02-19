import cv2
import numpy as np
import time
import torch
import torchvision.transforms as T
import matplotlib.pyplot as plt
from torchvision.models.detection import fasterrcnn_resnet50_fpn

from google.colab import drive
drive.mount('/content/drive')

image_path = "/5.jpg"

model = fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

def detect_faces_viola_jones(image_path):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    
    image = cv2.imread(image_path)
    if image is None:
        print("Mistake")
        return 0, 0, None
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    start_time = time.time()  
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    end_time = time.time()  
    
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 3)  
    
    execution_time = end_time - start_time
    return len(faces), execution_time, image

def detect_faces_cascade_rcnn(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print("Mistake")
        return 0, 0, None
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    transform = T.Compose([T.ToPILImage(), T.ToTensor()])
    image_tensor = transform(image_rgb).unsqueeze(0)
    
    start_time = time.time()  
    with torch.no_grad():
        predictions = model(image_tensor)
    end_time = time.time()  
    
    confidence_threshold = 0.8
    detected_faces = 0
    
    for i, score in enumerate(predictions[0]["scores"]):
        if score > confidence_threshold:
            x1, y1, x2, y2 = predictions[0]["boxes"][i].cpu().numpy().astype(int)
            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 3)  
            detected_faces += 1

    execution_time = end_time - start_time
    return detected_faces, execution_time, image

# 🔹 Запуск детекторов
faces_viola, time_viola, img_viola = detect_faces_viola_jones(image_path)
faces_cascade, time_cascade, img_cascade = detect_faces_cascade_rcnn(image_path)

# 🔹 Вывод результатов
print(f"Viola-Jones  {faces_viola} faces, Time: {time_viola:.4f} sec")
print(f"Cascade R-CNN  {faces_cascade} faces, Time: {time_cascade:.4f} sec")

# 🔹 Отображение изображений с найденными лицами
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
axes[0].imshow(cv2.cvtColor(img_viola, cv2.COLOR_BGR2RGB))
axes[0].set_title("Viola-Jones Detection")
axes[0].axis("off")

axes[1].imshow(cv2.cvtColor(img_cascade, cv2.COLOR_BGR2RGB))
axes[1].set_title("Cascade R-CNN Detection")
axes[1].axis("off")

plt.show()
