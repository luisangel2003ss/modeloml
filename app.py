import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf

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

# Título
st.title("Predicción de Derrames de Petróleo y Agua Producida")

# Entrada de usuario
st.subheader("Introduce los datos del incidente:")

release_cond = st.number_input("Cantidad de condensado liberado (release_cond) [en barriles]", min_value=0.0)
release_gas = st.number_input("Cantidad de gas liberado (release_gas) [en MCF]", min_value=0.0)

probable_cause = st.selectbox("Causa probable", list(traducciones_probable_cause.keys()))
type_operation = st.selectbox("Tipo de operación", list(traducciones_type_operation.keys()))

# Botón de predicción
if st.button("Predecir derrames"):
    try:
        # Traducción inversa
        probable_cause_orig = traducciones_probable_cause[probable_cause]
        type_operation_orig = traducciones_type_operation[type_operation]

        input_dict = {
            "release_cond": release_cond,
            "release_gas": release_gas,
            "probable_cause_edit": probable_cause_orig,
            "type_operation": type_operation_orig
        }

        df_input = pd.DataFrame([input_dict])

        # Preprocesamiento
        X_proc = preprocessor.transform(df_input)

        # Predicción
        pred_log = modelo_keras.predict(X_proc)
        pred_real = np.expm1(pred_log)

        st.success("Predicción completada:")
        st.write(f"🛢️ **Crude Oil estimado:** {pred_real[0][0]:.2f} barriles")
        st.write(f"💧 **Produced Water estimada:** {pred_real[0][1]:.2f} barriles")
    except Exception as e:
        st.error(f"Ocurrió un error en la predicción: {str(e)}")
