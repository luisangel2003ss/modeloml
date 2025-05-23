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

# Obtener columnas numéricas y categóricas del preprocesador
num_cols = preprocessor.transformers_[0][2]
cat_cols = preprocessor.transformers_[1][2]

st.title("Predicción de Derrames de Petróleo y Agua Producida")
st.subheader("Introduce los datos del incidente:")

input_data = {}

# Inputs para columnas numéricas
for col in num_cols:
    valor = st.number_input(f"{col} (numérico)", value=0.0)
    input_data[col] = valor

# Inputs para columnas categóricas
for col in cat_cols:
    # Para columnas con traducción conocidas, mostrar selectbox con opciones traducidas
    if col == "probable_cause_edit":
        opcion = st.selectbox("Causa probable", list(traducciones_probable_cause.keys()))
        input_data[col] = traducciones_probable_cause[opcion]
    elif col == "type_operation":
        opcion = st.selectbox("Tipo de operación", list(traducciones_type_operation.keys()))
        input_data[col] = traducciones_type_operation[opcion]
    else:
        # Si no hay traducción, permitir escribir texto libre o elegir de opciones fijas si tienes
        valor = st.text_input(f"{col} (categórico)", value="Unknown")
        input_data[col] = valor

if st.button("Predecir derrames"):
    try:
        # Armar DataFrame con un solo registro
        df_input = pd.DataFrame([input_data])

        # Preprocesamiento
        X_proc = preprocessor.transform(df_input)

        # Convertir a array si es sparse
        if hasattr(X_proc, "toarray"):
            X_proc = X_proc.toarray()

        # Predicción
        pred_log = modelo_keras.predict(X_proc)
        pred_real = np.expm1(pred_log)

        st.success("Predicción completada:")
        st.write(f"🛢️ **Crude Oil estimado:** {pred_real[0][0]:.2f} barriles")
        st.write(f"💧 **Produced Water estimada:** {pred_real[0][1]:.2f} barriles")

    except Exception as e:
        st.error(f"Ocurrió un error en la predicción: {str(e)}")
