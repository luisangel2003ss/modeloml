import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# Cargar modelo y preprocesador
modelo = load_model("modelo_multisalida.h5")
preprocessor = joblib.load("preprocessor.pkl")

# Opciones categóricas únicas (extraídas del dataset)
type_operations = ['PRODUCTION', 'COMPLETION', 'DRILLING', 'PIPELINE']
probable_causes = ['LEAK', 'HUMAN ERROR', 'LIGHTNING STRIKE', 'CORROSION']
districts = ['1', '2', '3', '4', '5', '6', '6E', '7C', '7B', '8', '8A', '9', '10']

# Función para realizar predicciones
def predecir(nuevos_datos_df):
    X_nuevo_proc = preprocessor.transform(nuevos_datos_df)
    if hasattr(X_nuevo_proc, "toarray"):
        X_nuevo_proc = X_nuevo_proc.toarray()

    pred_log = modelo.predict(X_nuevo_proc)
    pred = np.expm1(pred_log)
    return pd.DataFrame(pred, columns=["Crudo Derramado (barriles)", "Agua Derramada (barriles)"])

# Título de la aplicación
st.title("Predicción de Derrames de Crudo y Agua")

# Entradas del usuario
st.header("Ingresa los datos del incidente:")
cantidad_recuperada_crudo = st.number_input("Cantidad recuperada de crudo (barriles):", min_value=0.0)
cantidad_recuperada_agua = st.number_input("Cantidad recuperada de agua (barriles):", min_value=0.0)
tipo_operacion = st.selectbox("Tipo de operación", options=type_operations)
causa_probable = st.selectbox("Causa probable", options=probable_causes)
distrito = st.selectbox("Distrito", options=districts)

# Botón de predicción
if st.button("Predecir"):
    # Crear DataFrame con los datos ingresados
    nuevos_datos = pd.DataFrame({
        "type_operation": [tipo_operacion],

        "probable_cause": [causa_probable],

        "district_edit": [distrito],
        "recovery_crude_oil_edit": [cantidad_recuperada_crudo],
        "recovery_prod_water_edit": [cantidad_recuperada_agua],

    })
    
    # Realizar predicción
    try:
        predicciones = predecir(nuevos_datos)
        st.subheader("Resultados Predichos:")
        st.write(predicciones)
    except Exception as e:
        st.error(f"Error al realizar la predicción: {e}")
