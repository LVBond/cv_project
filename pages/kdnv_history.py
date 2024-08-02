import streamlit as st


st.title('История обучения')
st.caption('Серёжиной модельки')
st.divider()

st.subheader('Глава 1. Первая тренировка')
st.image('images/kudinov/params_1.jpg', use_column_width=True)
st.image('images/kudinov/metrics.png')
col = st.columns(2)
with col[0]:
    st.metric('Valid IoU', 0.63)
with col[1]:
    st.metric('Valid Loss', 0.52)
st.divider()

st.image('images/kudinov/pasha.png')
st.divider()

st.subheader('Глава 2. Вторая тренировка')
st.image('images/kudinov/params_2.jpg')
st.image('images/kudinov/metrics_2.png')
st.image('images/kudinov/meme.gif', use_column_width=True)
col = st.columns(2)
with col[0]:
    st.metric('Valid IoU', 0.73, delta=0.1)
with col[1]:
    st.metric('Valid Loss', 0.37, delta=-0.15)


st.divider()
st.image('images/kudinov/comp.jpg')

