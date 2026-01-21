import streamlit as st
import joblib
import os
import numpy as np
import pandas as pd

# Sayfa ayarları
st.set_page_config(
    page_title="Ev Fiyat Tahmin Sistemi",
    page_icon="🏠",
    layout="centered"
)

# Başlık
st.title("🏠 Ev Fiyat Tahmin Sistemi")
st.markdown("Makine öğrenmesi ile ev fiyatı tahmini yapan web uygulaması")

# Model yolu
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "results", "house_price_model.pkl")

# Model yükle
model = joblib.load(MODEL_PATH)

st.divider()

# Form alanı
st.subheader("📌 Ev Bilgilerini Giriniz")

col1, col2 = st.columns(2)

with col1:
    MedInc = st.slider("💰 Ortalama Gelir", 0.5, 15.0, 3.0)
    HouseAge = st.slider("🏗️ Ev Yaşı", 1, 50, 20)
    AveRooms = st.slider("🚪 Ortalama Oda Sayısı", 1.0, 10.0, 5.0)
    AveBedrms = st.slider("🛏️ Ortalama Yatak Odası", 0.5, 5.0, 1.0)

with col2:
    Population = st.slider("👥 Nüfus", 100, 10000, 1000)
    AveOccup = st.slider("🏠 Ortalama Doluluk", 1.0, 6.0, 3.0)
    Latitude = st.slider("🌍 Enlem", 32.0, 42.0, 34.0)
    Longitude = st.slider("🌍 Boylam", -125.0, -114.0, -118.0)

st.divider()

if st.button("🔮 Tahmin Et"):
    input_data = np.array([[MedInc, HouseAge, AveRooms, AveBedrms,
                            Population, AveOccup, Latitude, Longitude]])

    prediction = model.predict(input_data)[0]

    st.success(f"💵 Tahmini Ev Fiyatı: {prediction*100000:..0f} $")

    # Kullanılan girdileri tablo olarak göster
    df = pd.DataFrame(input_data, columns=[
        "MedInc", "HouseAge", "AveRooms", "AveBedrms",
        "Population", "AveOccup", "Latitude", "Longitude"
    ])

    st.subheader("📊 Girilen Değerler")
    st.dataframe(df)

    st.balloons()
