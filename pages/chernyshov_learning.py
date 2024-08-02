import streamlit as st
import pandas as pd


st.title('История обучения модели ResNet18')
st.divider()

st.subheader('Метрики')
st.write('Модель ResNet18 обучалась 100 эпох на предсказание 3 классов')

st.image('images/chernyshov/metrics.png', use_column_width=True)

data = {
    'Метрики': [
        'Loss', 'Classification Loss', 'BBox Loss',
        'Accuracy', 'IoU',
    ],
    'Train': [
        0.0037, 0.0001, 0.0036, 1.0000, 0.7237, 
    ],
    'Valid': [
        0.0084, 0.0036, 0.0048, 1.0000, 0.7346
    ]
}

metrics_df = pd.DataFrame(data)

st.table(metrics_df)
