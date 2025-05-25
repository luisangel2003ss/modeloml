import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# Funciones auxiliares
def cargar_modelos_y_preprocesador():
    preprocessor = joblib.load("preprocessor.pkl")
    modelo_crudo = load_model("modelo_crudo.h5")
    modelo_agua = load_model("modelo_agua.h5")
    return preprocessor, modelo_crudo, modelo_agua

def extraer_columnas_usadas(preprocessor):
    # Extrae columnas que espera el preprocesador (num + cat)
    columnas_numericas = preprocessor.transformers_[0][2]
    columnas_categoricas = preprocessor.transformers_[1][2]
    return list(columnas_numericas) + list(columnas_categoricas)

def validar_y_ordenar_columnas(df, columnas_esperadas):
    # Quita columnas extras y ordena según esperado
    faltantes = [c for c in columnas_esperadas if c not in df.columns]
    if faltantes:
        st.error(f"Faltan columnas requeridas: {faltantes}")
        st.stop()
    # Elimina columnas extras
    df = df.loc[:, df.columns.isin(columnas_esperadas)]
    # Ordena
    df = df[columnas_esperadas]
    return df

def predecir(df, preprocessor, modelo_crudo, modelo_agua, columnas_esperadas):
    df_validado = validar_y_ordenar_columnas(df, columnas_esperadas)
    X_proc = preprocessor.transform(df_validado)
    if hasattr(X_proc, "toarray"):
        X_proc = X_proc.toarray()
    pred_crudo = modelo_crudo.predict(X_proc).flatten()
    pred_agua = modelo_agua.predict(X_proc).flatten()
    # Evitar valores negativos
    pred_crudo = np.clip(pred_crudo, 0, None)
    pred_agua = np.clip(pred_agua, 0, None)
    return pred_crudo, pred_agua

# Interfaz Streamlit
st.title("Predicción de Derrames de Crudo y Agua")

uploaded_file = st.file_uploader("Sube un archivo CSV con datos de entrada", type=["csv"])

if uploaded_file is not None:
    try:
        df_input = pd.read_csv(uploaded_file)
        st.write("Datos cargados:")
        st.dataframe(df_input.head())

        preprocessor, modelo_crudo, modelo_agua = cargar_modelos_y_preprocesador()
        columnas_esperadas = extraer_columnas_usadas(preprocessor)

        pred_crudo, pred_agua = predecir(df_input, preprocessor, modelo_crudo, modelo_agua, columnas_esperadas)

        resultados = pd.DataFrame({
            "Predicción Crudo (bbl)": pred_crudo,
            "Predicción Agua (bbl)": pred_agua
        })
        st.write("Resultados de predicción:")
        st.dataframe(resultados)

    except Exception as e:
        st.error(f"Error al procesar el archivo o hacer predicciones: {e}")
