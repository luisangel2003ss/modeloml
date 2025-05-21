# -*- coding: utf-8 -*-
"""
Created on Wed May 21 15:01:05 2025

@author: RAUL SANCHEZ
"""

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.impute import SimpleImputer
import joblib

import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
import streamlit as st

# Configuración inicial para evitar problemas de caché
@st.cache_resource
def load_model_and_preprocessor():
    try:
        model = tf.keras.models.load_model("modelo_entrenado.h5", compile=False)
        preprocessor = joblib.load("preprocessor.pkl")
        return model, preprocessor, None
    except Exception as e:
        return None, None, str(e)

def main():
    st.title("Predicción de Derrames de Crudo y Agua")

    model, preprocessor, error = load_model_and_preprocessor()
    if error:
        st.error(f"Error al cargar los recursos:\n{error}\n\n"
                 "Verifica que los archivos 'modelo_entrenado.h5' y 'preprocessor.pkl' existan "
                 "y que las versiones de las librerías sean compatibles.")
        return

    # Entrada de usuario
    release_cond = st.number_input(
        "Condensado derramado (barriles)",
        min_value=0.0,
        max_value=150.0,
        value=10.0,
        help="Rango típico: 0-150 barriles"
    )

    release_gas = st.number_input(
        "Gas liberado (unidades)",
        min_value=0.0,
        max_value=10000.0,
        value=500.0,
        help="Rango típico: 0-10,000 unidades"
    )

    causas_dict = {
        'Corrosión': 'CORROSION',
        'Error humano': 'HUMAN ERROR',
        'Falla mecánica': 'MECHANICAL FAILURE',
        'Clima': 'WEATHER',
        'Falla de equipo': 'EQUIPMENT FAILURE'
    }
    causa_mostrada = st.selectbox("Causa probable", list(causas_dict.keys()))
    probable_cause = causas_dict[causa_mostrada]

    operaciones_dict = {
        'Producción': 'PRODUCTION',
        'Perforación': 'DRILLING',
        'Terminaciones': 'COMPLETIONS',
        'Inyección / Disposición': 'INJ/DISP',
        'Transporte': 'TRANSPORT'
    }
    operacion_mostrada = st.selectbox("Tipo de operación", list(operaciones_dict.keys()))
    operation_type = operaciones_dict[operacion_mostrada]

    if st.button("Predecir"):
        input_df = pd.DataFrame([{
            'release_cond': release_cond,
            'release_gas': release_gas,
            'probable_cause_edit': probable_cause,
            'type_operation': operation_type
        }])

        try:
            processed = preprocessor.transform(input_df)
            prediction = model.predict(processed)
            prediction = np.expm1(np.maximum(prediction, 0))

            st.success("Predicción exitosa:")
            st.write(f"Crudo estimado: {prediction[0][0]:.2f} barriles")
            st.write(f"Agua estimada: {prediction[0][1]:.2f} barriles")
        except Exception as e:
            st.error(f"Error durante la predicción:\n{str(e)}\n"
                     "Puede ser debido a valores fuera de rango o categorías no vistas en el entrenamiento.")

if __name__ == "__main__":
    main()
