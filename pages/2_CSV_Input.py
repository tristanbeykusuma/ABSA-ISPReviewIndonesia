import pickle
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd

with open('finalized_model_absa.pkl', 'rb') as f:
    vec_kartu, vec_internet, vec_harga, model_kartu, model_internet, model_harga = pickle.load(f)


uploaded_file = st.file_uploader("Masukkan CSV (rename kolom yang ingin diproses menjadi clean_teks)")
if uploaded_file is not None:
    #read csv
    df=pd.read_csv(uploaded_file)
    X_kartu = vec_kartu.transform(df['clean_teks'])
    X_internet = vec_internet.transform(df['clean_teks'])
    X_harga = vec_kartu.transform(df['clean_teks'])
    df['sentimen_kartu']=model_kartu.predict(X_kartu)
    df['sentimen_internet']=model_internet.predict(X_internet)
    df['sentimen_harga']=model_harga.predict(X_harga)
    st.dataframe(df, use_container_width=1)
else:
    st.warning('you need to upload a csv or excel file.')
