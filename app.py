import streamlit as st
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf

# Cargar modelo y preprocesador con manejo de errores y mensajes
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model('modelo_entrenado.h5', compile=False)
        preprocessor = joblib.load('preprocessor.pkl')
        return model, preprocessor, None
    except Exception as e:
        return None, None, str(e)

model, preprocessor, error = load_model()

if error:
    st.error(f"Error cargando el modelo o preprocesador: {error}")
else:
    st.success("Modelo y preprocesador cargados correctamente")

    st.title("Predicción de Derrames de Crudo y Agua de Producción")

    st.write("Ingrese los valores para realizar una predicción:")

    # Rango sugerido según el análisis previo
    release_cond = st.number_input("Condensado Derramado (barriles)", min_value=0.0, max_value=150.0, value=10.0)
    release_gas = st.number_input("Gas Liberado (unidades)", min_value=0.0, max_value=10000.0, value=500.0)

    probable_cause_edit = st.selectbox(
        "Causa Probable Verificada", 
        ['CORROSION', 'HUMAN ERROR', 'MECHANICAL FAILURE', 'WEATHER', 'EQUIPMENT FAILURE']
    )

    type_operation = st.selectbox(
        "Tipo de Operación", 
        ['PRODUCTION', 'DRILLING', 'COMPLETIONS', 'INJ/DISP', 'TRANSPORT']
    )

    if st.button("Predecir"):
        input_df = pd.DataFrame([{
            'release_cond': release_cond,
            'release_gas': release_gas,
            'probable_cause_edit': probable_cause_edit,
            'type_operation': type_operation
        }])

        try:
            processed_input = preprocessor.transform(input_df)
            prediction = model.predict(processed_input)
            prediction = np.expm1(np.maximum(prediction, 0))  # Log reverse + sin negativos

            st.success("Predicción completada:")
            st.write(f"Crudo derramado estimado: **{prediction[0][0]:.2f} barriles**")
            st.write(f"Agua de producción estimada: **{prediction[0][1]:.2f} barriles**")
        except Exception as e:
            st.error(f"Error en la predicción: {e}")

