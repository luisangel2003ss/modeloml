import streamlit as st
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model

# Cargar modelos y preprocesador
preprocessor = joblib.load("preprocessor.pkl")
modelo_crudo = load_model("modelo_crudo.h5")
modelo_agua = load_model("modelo_agua.h5")

st.title("Predicción de Derrames de Crudo y Agua")

# Entradas del usuario
crudo = st.number_input("Cantidad de crudo (barriles):", min_value=0.0, step=1.0)
gas = st.number_input("Cantidad de gas (Mcf):", min_value=0.0, step=1.0)

tipo_operacion = st.selectbox("Tipo de operación:", ["Producción", "Mantenimiento", "Perforación", "Transporte"])
tipo_error = st.selectbox("Tipo de error:", ["Error humano", "Corrosión", "Falla mecánica", "Otro"])
condicion = st.selectbox("Condición externa:", ["Perforación", "Clima adverso", "Normal"])

# Botón de predicción
if st.button("Predecir"):
    try:
        # Crear DataFrame con los datos ingresados
        entrada = pd.DataFrame([{
            "crudo": crudo,
            "gas": gas,
            "tipo_operacion": tipo_operacion,
            "tipo_error": tipo_error,
            "condicion": condicion
        }])

        # Preprocesar entrada
        X_proc = preprocessor.transform(entrada)
        if hasattr(X_proc, "toarray"):
            X_proc = X_proc.toarray()

        # Realizar predicciones
        pred_crudo = np.clip(modelo_crudo.predict(X_proc)[0][0], 0, None)
        pred_agua = np.clip(modelo_agua.predict(X_proc)[0][0], 0, None)

        # Mostrar resultados
        st.success(f"Predicción de crudo derramado: {pred_crudo:.2f} barriles")
        st.success(f"Predicción de agua producida derramada: {pred_agua:.2f} barriles")

    except Exception as e:
        st.error(f"Error en la predicción: {e}")
