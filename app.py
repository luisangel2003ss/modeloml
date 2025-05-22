import pandas as pd
import numpy as np
import joblib
import streamlit as st
import tensorflow as tf
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, OneHotEncoder

st.set_page_config(page_title="Predicción de Derrames", layout="centered")

@st.cache_resource
def load_model_and_preprocessor():
    try:
        model = tf.keras.models.load_model("modelo_entrenado.h5", compile=False)
        preprocessor = joblib.load("preprocessor.pkl")
        return model, preprocessor, None  # No hubo error
    except Exception as e:
        return None, None, str(e)  # Devuelves el mensaje de error


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

    st.markdown("### Ingrese los datos del evento:")

    # Entradas numéricas con botones +/-
    release_cond = st.number_input(
        "Condensado derramado (barriles)", min_value=0.0, max_value=100.0, step=0.01, value=0.0
    )
    release_gas = st.number_input(
        "Gas liberado (pies cúbicos)", min_value=0.0, max_value=5000.0, step=0.01, value=0.0
    )

    # Opciones de causa probable y tipo de operación
    causas_dict = {
    "Corrosión": "CORROSION",
    "Error humano": "HUMAN ERROR",
    "Falla de equipo": "EQUIPMENT FAILURE",
    "Sobrepresión": "OVERPRESSURE",
    "Falla estructural": "STRUCTURAL FAILURE",
    "Vandalismo": "VANDALISM"
}
    operaciones_dict = {
    "Producción": "PRODUCTION",
    "Perforación": "DRILLING",
    "Estimulación": "STIMULATION",
    "Mantenimiento": "MAINTENANCE"
}


    probable_cause_esp = st.selectbox("Causa probable", list(causas_dict.keys()))
    type_operation_esp = st.selectbox("Tipo de operación", list(operaciones_dict.keys()))

    probable_cause = causas_dict[probable_cause_esp]
    type_operation = operaciones_dict[type_operation_esp]

    if st.button("Predecir"):
        input_df = pd.DataFrame({
            "release_cond": [release_cond],
            "release_gas": [release_gas],
            "probable_cause_edit": [probable_cause],
            "type_operation": [type_operation]
    })

    model, preprocessor, error = load_model_and_preprocessor()

    if error:
        st.error(f"Error al cargar modelo o preprocesador: {error}")
        return

    prediction = predict(model, preprocessor, input_df)

    if prediction is not None:
        st.success("Predicción exitosa:")
        st.write(f"Crudo estimado: {prediction[0][0]:.2f} barriles")
        st.write(f"Agua estimada: {prediction[0][1]:.2f} barriles")


if __name__ == "__main__":
    main()
