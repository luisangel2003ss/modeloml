import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model

@st.cache_resource(allow_output_mutation=True)
def cargar_modelos():
    preprocessor = joblib.load("preprocessor.pkl")
    modelo_crudo = load_model("modelo_crudo.h5")
    modelo_agua = load_model("modelo_agua.h5")
    return preprocessor, modelo_crudo, modelo_agua

def predecir(df, preprocessor, modelo_crudo, modelo_agua):
    X_proc = preprocessor.transform(df)
    if hasattr(X_proc, "toarray"):
        X_proc = X_proc.toarray()
    pred_crudo = np.clip(modelo_crudo.predict(X_proc), 0, None).flatten()
    pred_agua = np.clip(modelo_agua.predict(X_proc), 0, None).flatten()
    return pred_crudo, pred_agua

def main():
    st.title("Predicción de Derrames de Crudo y Agua")

    uploaded_file = st.file_uploader("Sube archivo CSV con datos de entrada", type=["csv"])

    if uploaded_file is not None:
        df_entrada = pd.read_csv(uploaded_file)
        st.write("Datos de entrada:")
        st.dataframe(df_entrada)

        try:
            preprocessor, modelo_crudo, modelo_agua = cargar_modelos()

            # Validar columnas: eliminar columnas extras, ordenar si es necesario
            columnas_esperadas = preprocessor.get_feature_names_out()
            # get_feature_names_out puede no existir para ColumnTransformer con sklearn < 1.0,
            # en ese caso, usar método alternativo o asegurar sklearn actual.

            # Mejor validación básica:
            columnas_entrada = df_entrada.columns.tolist()
            faltantes = [c for c in columnas_esperadas if c not in columnas_entrada]
            if faltantes:
                st.error(f"Faltan columnas requeridas para preprocesar: {faltantes}")
                return
            # Aceptar solo columnas usadas
            df_entrada = df_entrada[columnas_esperadas]

            pred_crudo, pred_agua = predecir(df_entrada, preprocessor, modelo_crudo, modelo_agua)

            resultados = pd.DataFrame({
                "Predicción Crudo (bbl)": pred_crudo,
                "Predicción Agua (bbl)": pred_agua
            })
            st.write("Predicciones:")
            st.dataframe(resultados)

        except Exception as e:
            st.error(f"Error en la predicción: {e}")

if __name__ == "__main__":
    main()
