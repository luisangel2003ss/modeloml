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

# Obtener la lista de columnas numéricas y categóricas usadas en entrenamiento
# Se obtienen desde el preprocesador para asignar valores por defecto
columnas_numericas = preprocessor.transformers_[0][2]  # índices de columnas numéricas
columnas_categoricas = preprocessor.transformers_[1][2]  # índices de columnas categóricas

# Título
st.title("Predicción de Derrames de Petróleo y Agua Producida")

# Entrada de usuario
st.subheader("Introduce los datos del incidente:")

release_cond = st.number_input("Cantidad de condensado liberado (release_cond) [en barriles]", min_value=0.0)
release_gas = st.number_input("Cantidad de gas liberado (release_gas) [en MCF]", min_value=0.0)

probable_cause = st.selectbox("Causa probable", list(traducciones_probable_cause.keys()))
type_operation = st.selectbox("Tipo de operación", list(traducciones_type_operation.keys()))

if st.button("Predecir derrames"):
    try:
        # Traducción inversa de los valores categóricos
        probable_cause_orig = traducciones_probable_cause[probable_cause]
        type_operation_orig = traducciones_type_operation[type_operation]

        # Datos introducidos por el usuario
        input_data = {
            "release_cond": release_cond,
            "release_gas": release_gas,
            "probable_cause_edit": probable_cause_orig,
            "type_operation": type_operation_orig
        }

        # Obtener todas las columnas que el preprocesador espera (en orden y nombre)
        columnas_esperadas = preprocessor.feature_names_in_

        # Construir diccionario completo rellenando con valores por defecto
        input_data_completo = {}
        for col in columnas_esperadas:
            if col in input_data:
                input_data_completo[col] = input_data[col]
            else:
                # Rellenar numéricas con 0, categóricas con "Unknown"
                if col in columnas_numericas:
                    input_data_completo[col] = 0
                else:
                    input_data_completo[col] = "Unknown"

        # Crear DataFrame con todas las columnas necesarias
        df_input = pd.DataFrame([input_data_completo])

        # Preprocesar los datos
        X_proc = preprocessor.transform(df_input)
        if hasattr(X_proc, "toarray"):
            X_proc = X_proc.toarray()

        # Predecir con el modelo
        pred_log = modelo_keras.predict(X_proc)
        pred_real = np.expm1(pred_log)  # revertir transformación log(1+x)

        # Mostrar resultados
        st.success("Predicción completada:")
        st.write(f"🛢️ **Crude Oil estimado:** {pred_real[0][0]:.2f} barriles")
        st.write(f"💧 **Produced Water estimada:** {pred_real[0][1]:.2f} barriles")

    except Exception as e:
        st.error(f"Ocurrió un error en la predicción: {str(e)}")
