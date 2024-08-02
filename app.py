import streamlit as st

st.title('Супер классные нейросетки')
st.caption('От Серёжи Ч., Любы и Серёжи К.')
st.divider()

col1, col2, col3 = st.columns(3)

with col1:
    st.page_link('pages/chernyshov_model.py', label='Модель Серёжи Ч.', icon='🍆')
    st.page_link('pages/chernyshov_learning.py', label='Обучение', icon='💀')

with col2:
    st.page_link('pages/bond_model.py', label='Модель Любы.', icon='🧠')
    st.page_link('pages/bond_learning.py', label='Обучение', icon='ℹ️')

with col3:
    st.page_link('pages/kdnv_model.py', label='Модель Серёжи К.', icon='🌲')
    st.page_link('pages/kdnv_history.py', label='Инфа по модели', icon='👀')

st.divider()
# st.image('images/meme.gif')
