import streamlit as st
import joblib
import numpy as np

# =============================
# Load model
# =============================
model = joblib.load("house_price_model.pkl")

# =============================
# Judul Aplikasi
# =============================
st.title("🏠 Prediksi Harga Rumah di Seattle")
st.write("""
Aplikasi ini menggunakan model **Regresi Linier** untuk memprediksi harga rumah berdasarkan
luas rumah, jumlah kamar tidur, dan jumlah kamar mandi.
""")

# =============================
# Input dari pengguna
# =============================
col1, col2 = st.columns(2)

with col1:
    sqft_living = st.number_input("Luas Rumah (sqft)", min_value=100, max_value=10000, step=50, value=1800)

with col2:
    bedrooms = st.number_input("Jumlah Kamar Tidur", min_value=1, max_value=10, step=1, value=3)

bathrooms = st.slider("Jumlah Kamar Mandi", min_value=1.0, max_value=8.0, step=0.25, value=2.0)

# =============================
# Tombol Prediksi
# =============================
if st.button("Prediksi Harga Rumah"):
    # Masukkan input ke model
    features = np.array([[sqft_living, bedrooms, bathrooms]])
    prediction = model.predict(features)

    # =============================
    # Tampilkan hasil
    # =============================
    st.success(f"💰 Perkiraan Harga Rumah: **${prediction[0]:,.2f}**")

    st.caption("Model ini dilatih menggunakan dataset *King County House Sales* dari Kaggle.")

# =============================
# Footer
# =============================