import streamlit as st
import torch
from PIL import Image
import cv2
import numpy as np



# Загружаем модель YOLOv5
model = torch.hub.load('ultralytics/yolov5', 'custom', path='models/bond/best.pt')

# Словарь для отображения названий классов
class_names = {
    0: "Negative",
    1: "Positive"
}

# Функция для предсказания
def predict(image):
    results = model(image)
    
    # Получаем предсказания
    predictions = results.pred[0]
    
    # Фильтруем предсказания с высокой вероятностью
    filtered_predictions = predictions[predictions[:, 4] > 0.5]
    
    # Извлекаем информацию о предсказаниях
    class_ids = filtered_predictions[:, 5].long()
    bboxes = filtered_predictions[:, :4]
    
    return class_ids, bboxes

# Streamlit
st.title("Определение положительного или отрицательного класса")
uploaded_file = st.file_uploader("Загрузите изображение", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Загрузка изображения
    image = Image.open(uploaded_file)
    st.image(image, caption='Загруженное изображение', use_column_width=True)
    
    if st.button("Распознать"):
        # Предсказание
        class_ids, bboxes = predict(image)
        
        # Отображение результатов
        img = np.array(image)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        # Рисуем ограничивающие боксы и подписи
        for class_id, bbox in zip(class_ids, bboxes):
            x1, y1, x2, y2 = [int(coord) for coord in bbox]
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(img, class_names[class_id.item()], (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36, 255, 12), 2)
        
        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption='Результат', use_column_width=True)
...
