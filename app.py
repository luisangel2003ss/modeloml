import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf

# Traducciones fijas que me diste
traducciones_probable_cause = {
    "Hydraulic Failure": "Falla hidráulica",
    "Human Error": "Error humano",
    "Corrosion": "Corrosión"
}

traducciones_type_operation = {
    "Drilling": "Perforación",
    "Production": "Producción",
    "Maintenance": "Mantenimiento"
}

# Diccionarios inversos para mapear de español a categoría original
inv_trad_probable_cause = {v: k for k, v in traducciones_probable_cause.items()}
inv_trad_type_operation = {v: k for k, v in traducciones_type_operation.items()}

@st.cache_resource
def cargar_modelos():
    preprocessor = joblib.load("preprocessor.pkl")
    modelo_keras = tf.keras.models.load_model("modelo_entrenado.h5")
    return preprocessor, modelo_keras

preprocessor, modelo_keras = cargar_modelos()

st.title("Predicción de release_crude_oil y release_prod_wtr")

release_cond = st.number_input("release_cond", value=0.0)
release_gas = st.number_input("release_gas", value=0.0)

# Mostrar selectboxes con traducciones, guardando valor original
prob_causa_es = st.selectbox("Probable causa", list(traducciones_probable_cause.values()))
type_operacion_es = st.selectbox("Tipo de operación", list(traducciones_type_operation.values()))

# Mapear al valor original que espera el modelo
probable_cause_edit = inv_trad_probable_cause[prob_causa_es]
type_operation = inv_trad_type_operation[type_operacion_es]

if st.button("Predecir"):
    input_dict = {
        "release_cond": release_cond,
        "release_gas": release_gas,
        "probable_cause_edit": probable_cause_edit,
        "type_operation": type_operation
    }
    df_input = pd.DataFrame([input_dict])
    X_proc = preprocessor.transform(df_input)
    preds_log = modelo_keras.predict(X_proc)
    preds = np.expm1(preds_log)
    st.write("Predicción release_crude_oil y release_prod_wtr:")
    st.write(preds.flatten())
