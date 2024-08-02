import torch
from models.kudinov.model import UNet_orig
from models.kudinov.kdnv_preprocessing import preprocess
import streamlit as st
import requests
from io import BytesIO
from PIL import Image
from torchvision import transforms as T
import time


@st.cache_resource()
def load_model_1():
    model = UNet_orig(1)
    model.load_state_dict(torch.load('models/kudinov/model.pt', map_location=torch.device('cpu')))
    return model


@st.cache_resource()
def load_model_2():
    model = UNet_orig(1)
    model.load_state_dict(torch.load('models/kudinov/best_model.pt', map_location=torch.device('cpu')))
    return model


def resize(img):
    res = T.Compose([T.Resize((256, 256))])
    return res(img)


model_1 = load_model_1()
model_2 = load_model_2()


def predict(img):
    img = preprocess(img)
    with torch.inference_mode():
        if choice_1:
            pred = model_1(img)
        elif choice_2:
            pred = model_2(img)
    pred = torch.sigmoid(pred * -1)
    pred_mask = pred.squeeze().cpu().numpy()
    return pred_mask


def display(img):
    if choice_1 or choice_2:
        progress_bar = st.progress(0)
        start_time = time.time()

        pred_mask = predict(img)

        progress_bar.progress(100)
        end_time = time.time()
        execution_time = end_time - start_time
        col1, col2 = st.columns(2)

        st.write(f"Время выполнения: {execution_time:.2f} секунд")
        with col1:
            st.image(resize(img), caption='Оригинал', use_column_width=True)
        with col2:
            st.image(pred_mask, caption='Предикт', use_column_width=True)


st.title('Модель для поиска лесов на картах')
st.caption('От Серёжи')
st.divider()


uploaded_image = st.file_uploader("Кидай фото леса")
link = st.text_input('Кидай ссылку на фото леса')

if uploaded_image or link:
    col1, col2 = st.columns(2)
    with col1: choice_1 = st.button('Model_1')
    with col2: choice_2 = st.button('Model_2')

if uploaded_image:
    img = Image.open(uploaded_image)
    display(img)

if link:
    response = requests.get(link)
    img = Image.open(BytesIO(response.content))
    display(img)
