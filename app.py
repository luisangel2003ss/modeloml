import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
import matplotlib.pyplot as plt

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

# Carga modelo y preprocesador
preprocessor = joblib.load("preprocessor.pkl")
modelo_keras = tf.keras.models.load_model("modelo_entrenado.h5")

# Columnas del preprocesador
columnas_numericas = preprocessor.transformers_[0][2]
columnas_categoricas = preprocessor.transformers_[1][2]

# Título
st.title("🛢️ Predicción de Derrames de Crudo y Agua Producida")

# Entrada de usuario
st.subheader("Introduce los datos del incidente:")

release_cond = st.number_input(
    "Condensado Derramado (bbl)",
    min_value=0.0,
    max_value=137.0,
    value=10.0,
    help="Valores típicos: entre 1 y 137 barriles"
)

release_gas = st.number_input(
    "Gas Liberado (MCF)",
    min_value=0.0,
    max_value=9924.0,
    value=500.0,
    help="Valores típicos: entre 10 y 9,924 MCF"
)

# Advertencias por valores extremos
if release_cond < 1.0:
    st.warning("⚠️ Has ingresado un valor de condensado muy bajo. ¿Estás seguro de que no es cero por error?")
elif release_cond > 130.0:
    st.warning("⚠️ Estás ingresando un valor muy alto de condensado, cerca del máximo observado.")

if release_gas < 10.0:
    st.warning("⚠️ Has ingresado una cantidad de gas muy baja. Verifica que sea correcto.")
elif release_gas > 9500.0:
    st.warning("⚠️ Estás ingresando un valor muy alto de gas, cerca del límite máximo observado.")

# Selección categorías
probable_cause = st.selectbox("Causa probable", list(traducciones_probable_cause.keys()))
type_operation = st.selectbox("Tipo de operación", list(traducciones_type_operation.keys()))

if st.button("🔍 Predecir derrames"):
    try:
        # Traducción a inglés para el modelo
        probable_cause_orig = traducciones_probable_cause[probable_cause]
        type_operation_orig = traducciones_type_operation[type_operation]

        # Crear diccionario con datos de entrada
        input_data = {
            "release_cond": release_cond,
            "release_gas": release_gas,
            "probable_cause_edit": probable_cause_orig,
            "type_operation": type_operation_orig
        }

        # Construir dataframe con todas las columnas que espera el preprocesador
        columnas_esperadas = preprocessor.feature_names_in_
        input_data_completo = {}
        for col in columnas_esperadas:
            if col in input_data:
                input_data_completo[col] = input_data[col]
            elif col in columnas_numericas:
                input_data_completo[col] = 0
            else:
                input_data_completo[col] = "Unknown"

        df_input = pd.DataFrame([input_data_completo])

        # Preprocesar
        X_proc = preprocessor.transform(df_input)
        if hasattr(X_proc, "toarray"):  # Por si es sparse matrix
            X_proc = X_proc.toarray()

        # Predecir y revertir transformación log(1 + x)
        pred_log = modelo_keras.predict(X_proc)
        pred_real = np.expm1(pred_log)

        crude_pred = pred_real[0][0]
        water_pred = pred_real[0][1]

        # Controlar rangos para evitar negativos o valores absurdos
        crude_pred = max(0.0, min(crude_pred, 10000.0))
        water_pred = max(0.0, min(water_pred, 10000.0))

        # Mostrar resultados
        st.success("✅ Predicción completada:")
        st.write(f"🛢️ **Crudo Derramado estimado:** {crude_pred:.2f} barriles")
        st.write(f"💧 **Agua Producida estimada:** {water_pred:.2f} barriles")

        # Gráfico para crudo derramado
        fig_crudo, ax_crudo = plt.subplots()
        ax_crudo.bar(["Crudo Derramado"], [crude_pred], color="saddlebrown")
        ax_crudo.set_ylabel("Barriles")
        ax_crudo.set_title("Predicción de Crudo Derramado")
        st.pyplot(fig_crudo)

        # Gráfico para agua producida
        fig_agua, ax_agua = plt.subplots()
        ax_agua.bar(["Agua Producida"], [water_pred], color="skyblue")
        ax_agua.set_ylabel("Barriles")
        ax_agua.set_title("Predicción de Agua Producida")
        st.pyplot(fig_agua)

    except Exception as e:
        st.error(f"❌ Ocurrió un error en la predicción: {str(e)}")
