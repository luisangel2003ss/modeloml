import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
import matplotlib.pyplot as plt

# Diccionarios de traducción
causas_dict = {
    'Corrosión': 'CORROSION',
    'Error humano': 'HUMAN ERROR',
    'Falla mecánica': 'MECHANICAL FAILURE',
    'Clima': 'WEATHER',
    'Falla de equipo': 'EQUIPMENT FAILURE'
}

operaciones_dict = {
    'Producción': 'PRODUCTION',
    'Perforación': 'DRILLING',
    'Terminaciones': 'COMPLETIONS',
    'Inyección / Disposición': 'INJ/DISP',
    'Transporte': 'TRANSPORT'
}

# Cargar modelo y preprocesador
preprocessor = joblib.load("preprocessor.pkl")
modelo_keras = tf.keras.models.load_model("modelo_entrenado.h5")

# Columnas del preprocesador
columnas_numericas = preprocessor.transformers_[0][2]
columnas_categoricas = preprocessor.transformers_[1][2]

# Título
st.title("🛢️ Predicción de Derrames de Crudo y Agua Producida")

# Entrada de usuario
st.subheader("Introduce los datos del incidente:")
col1, col2 = st.columns(2)

with col1:
    release_cond = st.number_input("Condensado Derramado (bbl)", min_value=0.0, max_value=150.0, value=10.0)
    causa_mostrada = st.selectbox("Causa probable", options=list(causas_dict.keys()))
    probable_cause = causas_dict[causa_mostrada]
    recovery_crude = st.number_input("Recuperación de Crudo (bbl)", min_value=0.0, max_value=10000.0, value=0.0)

with col2:
    release_gas = st.number_input("Gas Liberado (unidades)", min_value=0.0, max_value=10000.0, value=500.0)
    operacion_mostrada = st.selectbox("Tipo de operación", options=list(operaciones_dict.keys()))
    type_operation = operaciones_dict[operacion_mostrada]
    recovery_water = st.number_input("Recuperación de Agua Producida (bbl)", min_value=0.0, max_value=10000.0, value=0.0)

# Botón de predicción
if st.button("🔍 Predecir derrames"):
    try:
        # Diccionario de entrada
        input_data = {
            "release_cond": release_cond,
            "release_gas": release_gas,
            "probable_cause_edit": probable_cause,
            "type_operation": type_operation,
            "recovery_crude_oil_edit": recovery_crude,
            "recovery_prod_water_edit": recovery_water
        }

        # Completar columnas faltantes
        columnas_esperadas = preprocessor.feature_names_in_
        input_completo = {}
        for col in columnas_esperadas:
            if col in input_data:
                input_completo[col] = input_data[col]
            elif col in columnas_numericas:
                input_completo[col] = 0.0
            else:
                input_completo[col] = "Unknown"

        df_input = pd.DataFrame([input_completo])

        # Preprocesar
        X_proc = preprocessor.transform(df_input)
        if hasattr(X_proc, "toarray"):
            X_proc = X_proc.toarray()

        # Predecir
        y_pred = modelo_keras.predict(X_proc)
        y_pred = np.clip(y_pred, 0, None)  # Solo positivos

        crude_pred = float(y_pred[0][0])
        water_pred = float(y_pred[0][1])

        # Mostrar resultados
        st.success("✅ Predicción completada:")
        st.write(f"🛢️ **Crudo Derramado estimado:** {crude_pred:.2f} barriles")
        st.write(f"💧 **Agua Producida estimada:** {water_pred:.2f} barriles")

        # Gráficos de barras
        fig_bar, ax_bar = plt.subplots()
        ax_bar.bar(["Crudo Derramado", "Agua Producida"], [crude_pred, water_pred], color=["saddlebrown", "skyblue"])
        ax_bar.set_ylabel("Barriles")
        ax_bar.set_title("Predicción de Derrames")
        st.pyplot(fig_bar)

        # Histograma (aunque con un solo dato no aporta mucho, queda preparado para batchs)
        fig_hist, axs = plt.subplots(1, 2, figsize=(10, 4))
        axs[0].hist([crude_pred], bins=5, color="saddlebrown", alpha=0.7)
        axs[0].set_title("Distribución - Crudo Derramado")
        axs[0].set_xlabel("Barriles")
        axs[0].set_ylabel("Frecuencia")

        axs[1].hist([water_pred], bins=5, color="skyblue", alpha=0.7)
        axs[1].set_title("Distribución - Agua Producida")
        axs[1].set_xlabel("Barriles")
        axs[1].set_ylabel("Frecuencia")

        plt.tight_layout()
        st.pyplot(fig_hist)

    except Exception as e:
        st.error(f"❌ Ocurrió un error: {str(e)}")
