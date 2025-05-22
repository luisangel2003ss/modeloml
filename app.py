import pandas as pd
import numpy as np
import joblib
import streamlit as st
import tensorflow as tf

st.set_page_config(page_title="Predicción de Derrames", layout="centered")

@st.cache_resource
def load_model_and_preprocessor():
    try:
        model = tf.keras.models.load_model("modelo_entrenado.h5", compile=False)
        preprocessor = joblib.load("preprocessor.pkl")
        return model, preprocessor, None
    except Exception as e:
        return None, None, str(e)


def predict(model, preprocessor, data):
    try:
        processed = preprocessor.transform(data)
        pred = model.predict(processed)
        pred = np.expm1(pred)  # Deshacer log1p si fue usado en entrenamiento
        pred = np.maximum(pred, 0)  # Asegurar valores positivos
        return pred
    except Exception as e:
        st.error(f"Error durante la predicción: {e}")
        return None


def main():
    st.title("Predicción de Derrames de Crudo y Agua")

    # Inputs
    release_cond = st.number_input("Condensado derramado (barriles)", min_value=0.0, max_value=100.0, step=0.01, value=0.0)
    release_gas = st.number_input("Gas liberado (pies cúbicos)", min_value=0.0, max_value=5000.0, step=0.01, value=0.0)

    # Diccionarios con valores reales (ejemplo)
    causas_dict = {
        "Falla mecánica": "mechanical_failure",
        "Error humano": "human_error",
        "Condiciones climáticas": "weather_conditions",
    }
    operaciones_dict = {
        "Producción": "production",
        "Mantenimiento": "maintenance",
        "Prueba": "testing",
    }

    probable_cause_esp = st.selectbox("Causa probable", list(causas_dict.keys()))
    type_operation_esp = st.selectbox("Tipo de operación", list(operaciones_dict.keys()))

    probable_cause = causas_dict[probable_cause_esp]
    type_operation = operaciones_dict[type_operation_esp]

    model, preprocessor, error = load_model_and_preprocessor()

    if error:
        st.error(f"Error al cargar modelo o preprocesador: {error}")
        return

    if st.button("Predecir"):
        # Aquí confirmar que los nombres de columnas coinciden con los usados en preprocesador
        input_df = pd.DataFrame({
            "release_cond": [release_cond],
            "release_gas": [release_gas],
            "probable_cause_edit": [probable_cause],  # Verifica nombre exacto
            "type_operation": [type_operation]
        })

        prediction = predict(model, preprocessor, input_df)

        if prediction is not None:
            st.success("Predicción exitosa:")
            # Confirmar estructura de prediction; asumo (1, 2) => [ [crudo, agua] ]
            st.write(f"Crudo estimado: {prediction[0][0]:.2f} barriles")
            st.write(f"Agua estimada: {prediction[0][1]:.2f} barriles")


if __name__ == "__main__":
    main()
