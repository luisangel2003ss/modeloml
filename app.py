import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

# Cargar modelos
modelo_crudo = load_model("modelo_crudo.h5")
modelo_agua = load_model("modelo_agua.h5")

st.title("Predicción de Derrames")

# Entradas numéricas
cantidad_crudo = st.number_input("Cantidad de crudo (barriles)", min_value=0.0, step=0.1)
cantidad_gas = st.number_input("Cantidad de gas (m³)", min_value=0.0, step=0.1)

# Selección de tipo de operación
tipo_operacion = st.selectbox("Tipo de operación", ["Extracción", "Transporte", "Almacenamiento"])

# Selección de tipo de error
tipo_error = st.selectbox("Tipo de error", ["Error humano", "Corrosión", "Falla mecánica", "Otro"])

# Mapeo de variables categóricas a valores numéricos (one-hot encoding)
operacion_map = {
    "Extracción": [1, 0, 0],
    "Transporte": [0, 1, 0],
    "Almacenamiento": [0, 0, 1]
}

error_map = {
    "Error humano": [1, 0, 0, 0],
    "Corrosión": [0, 1, 0, 0],
    "Falla mecánica": [0, 0, 1, 0],
    "Otro": [0, 0, 0, 1]
}

# Construir el vector de entrada
entrada = [cantidad_crudo, cantidad_gas] + operacion_map[tipo_operacion] + error_map[tipo_error]
entrada_array = np.array([entrada])

# Realizar predicciones
if st.button("Predecir"):
    pred_crudo = modelo_crudo.predict(entrada_array)
    pred_agua = modelo_agua.predict(entrada_array)

    # Mostrar resultados
    st.write("Predicción de derrame de crudo:", np.clip(pred_crudo, 0, None).flatten()[0])
    st.write("Predicción de derrame de agua:", np.clip(pred_agua, 0, None).flatten()[0])
