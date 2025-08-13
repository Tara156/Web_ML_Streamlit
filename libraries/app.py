from pickle import load
import streamlit as st
import numpy as np
import os

# 📦 Importar modelo (ruta relativa)

# model = load(open("models/xg_boost_diabetes_v1.sav", "rb"))
model = load(open("models/xg_boost_diabetes_v1.sav", "rb"))




# 🛠 Configuración de página
st.set_page_config(page_title="Predicción de Diabetes", page_icon="🩺", layout="wide")

with st.container():
    st.title("🩺 Predicción de Diabetes")
    st.write("Esta aplicación predice si una persona tiene riesgo de diabetes en base a parámetros médicos.")

with st.container():
    st.subheader("🔍 Ingresa las características del paciente")
    col1 = st.columns(1)[0]
    with col1:
        pregnancies = st.slider("🤰 Embarazos", 0, 20, 1)
        glucose = st.slider("🍬 Glucosa", 0, 200, 120)
        blood_pressure = st.slider("💓 Presión Sanguínea", 0, 140, 70)
        bmi = st.slider("⚖️ Índice de Masa Corporal (BMI)", 0.0, 70.0, 25.0, 0.1)
        diabetes_pedigree = st.slider("📊 Diabetes Pedigree Function", 0.0, 2.5, 0.5, 0.01)
        age = st.slider("🎂 Edad", 0, 120, 30)

with st.container():
    st.markdown("---")
    if st.button("✨ Predecir"):
        input_data = np.array([[pregnancies, glucose, blood_pressure, bmi, diabetes_pedigree, age]])
        prediction = model.predict(input_data)[0]

        if prediction == 1:
            st.error("⚠️ El modelo predice que **tiene riesgo de diabetes**.")
        else:
            st.success("✅ El modelo predice que **NO tiene riesgo de diabetes**.")
