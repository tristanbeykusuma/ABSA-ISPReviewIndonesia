import pickle
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer

with open('finalized_model_absa.pkl', 'rb') as f:
    vec_kartu, vec_internet, vec_harga, model_kartu, model_internet, model_harga = pickle.load(f)


text_input = st.text_input(
    "Masukkan teks ulasan yang diinginkan 👇"
)

if text_input:
    st.write("You entered: ", text_input)

x_harga=vec_harga.transform([text_input]).toarray()
x_internet=vec_internet.transform([text_input]).toarray()
x_kartu=vec_kartu.transform([text_input]).toarray()

st.write("Aspect Kartu", 'Positive' if model_kartu.predict(x_kartu) == 1 else 'Negative')
st.write("Aspect Internet", 'Positive' if model_internet.predict(x_internet) == 1 else 'Negative')
st.write("Aspect Harga", 'Positive' if model_harga.predict(x_harga) == 1 else 'Negative')
