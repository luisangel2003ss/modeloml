import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# Cargar preprocesador y modelos
@st.cache_resource
def cargar_modelos():
    preprocessor = joblib.load("preprocessor.pkl")
    modelo_crudo = load_model("modelo_crudo.h5")
    modelo_agua = load_model("modelo_agua.h5")
    return preprocessor, modelo_crudo, modelo_agua

preprocessor, modelo_crudo, modelo_agua = cargar_modelos()

def predecir(df_entrada):
    X_proc = preprocessor.transform(df_entrada)
    if hasattr(X_proc, "toarray"):  # si es sparse
        X_proc = X_proc.toarray()
    pred_crudo = np.clip(modelo_crudo.predict(X_proc), 0, None).flatten()
    pred_agua = np.clip(modelo_agua.predict(X_proc), 0, None).flatten()
    return pred_crudo, pred_agua

st.title("Predicción de Derrames de Crudo y Agua")

uploaded_file = st.file_uploader("Carga un archivo CSV con los datos de entrada", type=["csv"])

if uploaded_file is not None:
    df_entrada = pd.read_csv(uploaded_file)
    st.write("Datos cargados:")
    st.dataframe(df_entrada.head())

    try:
        pred_crudo, pred_agua = predecir(df_entrada)
        resultados = df_entrada.copy()
        resultados["Predicción Crudo (bbl)"] = pred_crudo
        resultados["Predicción Agua (bbl)"] = pred_agua

        st.write("Resultados de la predicción:")
        st.dataframe(resultados)

        # Opcional: descargar resultados
        csv_result = resultados.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Descargar resultados como CSV",
            data=csv_result,
            file_name='resultados_prediccion.csv',
            mime='text/csv'
        )

    except Exception as e:
        st.error(f"Error en la predicción: {e}")

else:
    st.info("Por favor, carga un archivo CSV para hacer predicciones.")
