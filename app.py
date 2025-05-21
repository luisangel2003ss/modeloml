import pandas as pd
import numpy as np
import joblib
import streamlit as st
import tensorflow as tf

st.set_page_config(page_title="Predicción de Derrames", layout="centered")

@st.cache_resource
def load_model_and_preprocessor():
    model = tf.keras.models.load_model("modelo_entrenado.h5", compile=False)
    preprocessor = joblib.load("preprocessor.pkl")
    return model, preprocessor

def predict(model, preprocessor, data):
    try:
        processed = preprocessor.transform(data)
        pred = model.predict(processed)
        pred = np.expm1(pred)
        pred = np.maximum(pred, 0)
        return pred
    except Exception as e:
        st.error(f"Error durante la predicción: {e}")
        return None

def main():
    st.title("Predicción de Derrames de Crudo y Agua")

    st.markdown("### Ingrese los datos manualmente (se validarán los rangos):")

    # Entradas como texto
    release_cond = st.text_input("Condensado derramado (0 - 100)")
    release_gas = st.text_input("Gas liberado (0 - 5000)")

    probable_cause = st.text_input("Causa probable (ej: CORROSION, HUMAN ERROR)").upper()
    type_operation = st.text_input("Tipo de operación (ej: PRODUCTION, DRILLING)").upper()

    if st.button("Predecir"):
        try:
            cond = float(release_cond)
            gas = float(release_gas)

            if not (0 <= cond <= 100):
                st.error("El valor de condensado debe estar entre 0 y 100.")
                return
            if not (0 <= gas <= 5000):
                st.error("El valor de gas debe estar entre 0 y 5000.")
                return

            input_df = pd.DataFrame({
                "release_cond": [cond],
                "release_gas": [gas],
                "probable_cause_edit": [probable_cause],
                "type_operation": [type_operation]
            })

            model, preprocessor = load_model_and_preprocessor()
            prediction = predict(model, preprocessor, input_df)

            if prediction is not None:
                st.success("Predicción exitosa:")
                st.write(f"Crudo estimado: {prediction[0][0]:.2f} barriles")
                st.write(f"Agua estimada: {prediction[0][1]:.2f} barriles")

        except ValueError:
            st.error("Ingrese valores numéricos válidos para 'Condensado' y 'Gas'.")

if __name__ == "__main__":
    main()
