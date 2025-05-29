import streamlit as st
import pandas as pd
import numpy as np
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

# Load model dan encoder
model = joblib.load("naive_bayes_weather_model.pkl")
le = joblib.load("weather_label_encoder.pkl")

# Load dataset bawaan
df = pd.read_csv("1. Weather Data.csv")

# ============================
st.set_page_config(page_title="Prediksi Cuaca", layout="wide")
st.title("🌦️ Aplikasi Prediksi Cuaca - Naive Bayes")

menu = st.sidebar.radio("Navigasi", ["Deskripsi Data", "Prediksi", "Visualisasi"])

# ============================
# 📊 DESKRIPSI DATA
if menu == "Deskripsi Data":
    st.header("📊 Deskripsi Dataset Cuaca")

    st.subheader("5 Baris Pertama Data:")
    st.dataframe(df.head())

    st.subheader("Statistik Deskriptif:")
    st.write(df.describe())

    st.subheader("Informasi Kolom:")
    info_df = pd.DataFrame({
        "Kolom": df.columns,
        "Tipe Data": df.dtypes.astype(str),
        "Jumlah Null": df.isnull().sum()
    })
    st.write(info_df)

# ============================
# 🔍 PREDIKSI
elif menu == "Prediksi":
    st.header("🔍 Prediksi Cuaca")
    st.write("Masukkan fitur cuaca berikut:")

    temp = st.slider("Temperatur (°C)", -30.0, 40.0, 20.0)
    dew = st.slider("Titik Embun (°C)", -30.0, 30.0, 10.0)
    hum = st.slider("Kelembapan Relatif (%)", 0, 100, 70)
    wind = st.slider("Kecepatan Angin (km/h)", 0, 100, 10)
    vis = st.slider("Jarak Pandang (km)", 0.0, 50.0, 20.0)
    press = st.slider("Tekanan Udara (kPa)", 95.0, 105.0, 101.0)
    hour = st.slider("Jam", 0, 23, 12)
    month = st.slider("Bulan", 1, 12, 6)
    day = st.slider("Hari", 1, 31, 15)
    weekday = st.slider("Hari ke-", 0, 6, 2)

    if st.button("Prediksi"):
        input_data = np.array([[temp, dew, hum, wind, vis, press, hour, month, day, weekday]])
        prediction = model.predict(input_data)
        label = le.inverse_transform(prediction)
        st.success(f"🌤️ Prediksi Cuaca: **{label[0]}**")

# ============================
# 📈 VISUALISASI
elif menu == "Visualisasi":
    st.header("📈 Visualisasi Cuaca")

    if 'Weather' in df.columns:
        st.subheader("Distribusi Kategori Cuaca")
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.countplot(data=df, x='Weather', order=df['Weather'].value_counts().index, ax=ax)
        plt.xticks(rotation=45)
        st.pyplot(fig)
    else:
        st.warning("Kolom 'Weather' tidak ditemukan dalam dataset.")
