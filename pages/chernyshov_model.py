import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2
import requests
from io import BytesIO

# Загружаем модель
class ResNet18Model(nn.Module):
    def __init__(self, num_classes):
        super(ResNet18Model, self).__init__()
        self.resnet = models.resnet18(pretrained=True)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes + 4)

    def forward(self, x):
        x = self.resnet(x)
        class_logits = x[:, :3]  
        bbox_coords = x[:, 3:]   
        return class_logits, bbox_coords

# Инициализация модели
num_classes = 3  
model = ResNet18Model(num_classes=num_classes)
model.load_state_dict(torch.load('models/chernyshov/model.pth', map_location=torch.device('cpu')))
model.eval()

# Словарь для отображения названий классов
class_names = {
    0: "Cucumber",
    1: "Eggplant",
    2: "Mushroom"
}

# Функция для предсказания
def predict(image):
    transform = transforms.Compose([
        transforms.Resize((227, 227)),
        transforms.ToTensor(),
    ])
    
    image = transform(image).unsqueeze(0) 
    with torch.no_grad():
        class_logits, predicted_bboxes = model(image)
        
    return class_logits, predicted_bboxes

# Streamlit
st.title("Привет! Я могу помочь тебе отличить гриб от баклажана или огурца")
uploaded_file = st.file_uploader("Не знаю зачем тебе это, но загрузи изображение...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Загрузка изображения
    image = Image.open(uploaded_file)
    st.image(image, caption='Успешно', use_column_width=True)
    
    if st.button("Узнать, что изображено на картинке"):
        class_logits, predicted_bboxes = predict(image)
        
        # Обработка предсказаний
        predicted_class_index = class_logits.argmax(dim=1).item()
        predicted_class_name = class_names[predicted_class_index]  # Получаем название класса
        bbox = predicted_bboxes.squeeze().numpy() * 227  # Масштабируем обратно
        
        # Отображение результатов
        img = np.array(image)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        # Рисуем ограничивающий бокс
        cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 0, 0), 2)
        
        # Добавляем текст с названием класса
        cv2.putText(img, predicted_class_name, (int(bbox[0]), int(bbox[1]) - 10), 
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption='Predicted Image', use_column_width=True)
        st.write(f"Это: {predicted_class_name}")  # Выводим название класса
