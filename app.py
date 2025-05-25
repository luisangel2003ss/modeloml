import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model

@st.cache_resource
def cargar_modelos():
    preprocessor = joblib.load("preprocessor.pkl")
    modelo_crudo = load_model("modelo_crudo.h5")
    modelo_agua = load_model("modelo_agua.h5")
    return preprocessor, modelo_crudo, modelo_agua

def predecir(df_entrada, preprocessor, modelo_crudo, modelo_agua):
    X_proc = preprocessor.transform(df_entrada)
    if hasattr(X_proc, "toarray"):  # si es matriz dispersa
        X_proc = X_proc.toarray()
    pred_crudo = np.clip(modelo_crudo.predict(X_proc), 0, None).flatten()
    pred_agua = np.clip(modelo_agua.predict(X_proc), 0, None).flatten()
    return pred_crudo, pred_agua

def main():
    st.title("Predicción de Derrames de Crudo y Agua")

    preprocessor, modelo_crudo, modelo_agua = cargar_modelos()

    uploaded_file = st.file_uploader("Carga un archivo CSV con los datos de entrada", type=["csv"])

    if uploaded_file is not None:
        df_input = pd.read_csv(uploaded_file)

        st.write("Datos cargados:")
        st.dataframe(df_input.head())

        if st.button("Predecir"):
            try:
                pred_crudo, pred_agua = predecir(df_input, preprocessor, modelo_crudo, modelo_agua)

                resultados = df_input.copy()
                resultados["Predicción Crudo"] = pred_crudo
                resultados["Predicción Agua"] = pred_agua

                st.write("Resultados de la predicción:")
                st.dataframe(resultados)

            except Exception as e:
                st.error(f"Error en la predicción: {e}")

if __name__ == "__main__":
    main()
