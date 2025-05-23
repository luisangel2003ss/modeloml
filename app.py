import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
import plotly.express as px

# Traducciones
traducciones_probable_cause = {
    "Falla hidráulica": "Hydraulic Failure",
    "Error humano": "Human Error",
    "Corrosión": "Corrosion"
}

traducciones_type_operation = {
    "Perforación": "Drilling",
    "Producción": "Production",
    "Mantenimiento": "Maintenance"
}

# Carga el preprocesador y modelo
preprocessor = joblib.load("preprocessor.pkl")
modelo_keras = tf.keras.models.load_model("modelo_entrenado.h5")

# Obtener columnas usadas en entrenamiento
columnas_numericas = preprocessor.transformers_[0][2]
columnas_categoricas = preprocessor.transformers_[1][2]

# Título
st.title("🔍 Predicción de Derrames de Petróleo y Agua Producida")

st.subheader("🧾 Introduce los datos del incidente:")

# Entradas numéricas con validaciones
release_cond = st.number_input(
    "🛢️ Condensado Derramado (bbl)",
    min_value=0.0,
    max_value=137.0,
    value=10.0,
    help="Valores típicos: entre 1 y 137 barriles"
)
if release_cond < 1.0:
    st.warning("⚠️ Has ingresado un valor de condensado muy bajo. ¿Estás seguro de que no es cero por error?")
elif release_cond > 130.0:
    st.warning("⚠️ Estás ingresando un valor muy alto de condensado, cerca del máximo observado.")

release_gas = st.number_input(
    "🔥 Gas Liberado (MCF)",
    min_value=0.0,
    max_value=9924.0,
    value=500.0,
    help="Valores típicos: entre 10 y 9,924 MCF"
)
if release_gas < 10.0:
    st.warning("⚠️ Has ingresado una cantidad de gas muy baja. Verifica que sea correcto.")
elif release_gas > 9500.0:
    st.warning("⚠️ Estás ingresando
