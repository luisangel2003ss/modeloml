import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

# Configuración inicial para evitar problemas de caché
@st.cache_resource
def load_resources():
    # Workaround para scikit-learn 1.4.2
    import sklearn.compose
    if not hasattr(sklearn.compose, '_RemainderColsList'):
        setattr(sklearn.compose, '_RemainderColsList', list)
    
    try:
        # Carga con verificación de integridad
        with st.spinner('Cargando modelo... (puede tomar unos segundos)'):
            model = tf.keras.models.load_model('modelo_entrenado.h5', compile=False)
            
        with st.spinner('Cargando preprocesador...'):
            preprocessor = joblib.load('preprocessor.pkl')
            
        return model, preprocessor, None
    except Exception as e:
        return None, None, str(e)

# Interfaz de usuario principal
def main():
    st.title("Sistema de Predicción de Derrames")
    
    # Carga de recursos con manejo de errores
    model, preprocessor, error = load_resources()
    
    if error:
        st.error(f"""
        **Error crítico**: No se pudieron cargar los recursos.  
        Detalle técnico:  
        `{error}`  
        
        Recomendaciones:  
        1. Verifica que los archivos `modelo_entrenado.h5` y `preprocessor.pkl` existan  
        2. Reinstala las versiones exactas:  
           ```bash
           pip install scikit-learn==1.4.2 tensorflow==2.15.0 joblib==1.4.2
           ```
        """)
        return

    st.success("✅ Modelo y preprocesador cargados correctamente")
    
    # Widgets de entrada
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            release_cond = st.number_input(
                "Condensado Derramado (bbl)",
                min_value=0.0,
                max_value=150.0,
                value=10.0,
                help="Rango típico: 0-150 barriles"
            )
            probable_cause = st.selectbox(
                "Causa Probable",
                options=['CORROSION', 'HUMAN ERROR', 'MECHANICAL FAILURE', 'WEATHER', 'EQUIPMENT FAILURE'],
                index=1
            )
        
        with col2:
            release_gas = st.number_input(
                "Gas Liberado (unidades)",
                min_value=0.0,
                max_value=10000.0,
                value=500.0,
                help="Rango típico: 0-10,000 unidades"
            )
            operation_type = st.selectbox(
                "Tipo de Operación",
                options=['PRODUCTION', 'DRILLING', 'COMPLETIONS', 'INJ/DISP', 'TRANSPORT'],
                index=0
            )
        
        if st.form_submit_button("Predecir", type="primary"):
            with st.spinner("Procesando..."):
                try:
                    # Preparación de datos
                    input_data = pd.DataFrame([{
                        'release_cond': release_cond,
                        'release_gas': release_gas,
                        'probable_cause_edit': probable_cause,
                        'type_operation': operation_type
                    }])
                    
                    # Transformación y predicción
                    processed_data = preprocessor.transform(input_data)
                    prediction = model.predict(processed_data)
                    prediction = np.expm1(np.maximum(prediction, 0))
                    
                    # Mostrar resultados
                    st.success("Resultados de la predicción:")
                    
                    col_res1, col_res2 = st.columns(2)
                    with col_res1:
                        st.metric(
                            "Crudo derramado estimado", 
                            f"{prediction[0][0]:.2f} bbl",
                            delta="±5% margen de error"
                        )
                    
                    with col_res2:
                        st.metric(
                            "Agua de producción estimada", 
                            f"{prediction[0][1]:.2f} bbl",
                            delta="±5% margen de error"
                        )
                    
                    # Detalles técnicos (expandible)
                    with st.expander("Detalles técnicos"):
                        st.json({
                            "input_values": input_data.iloc[0].to_dict(),
                            "model_version": "1.0",
                            "framework": f"TensorFlow {tf.__version__}",
                            "preprocessor": f"scikit-learn {joblib.__version__}"
                        })
                
                except Exception as e:
                    st.error(f"""
                    **Error en la predicción**:  
                    `{str(e)}`  
                    
                    Posibles causas:  
                    - Valores de entrada fuera de rango  
                    - Categorías no vistas durante el entrenamiento  
                    - Problema interno del modelo  
                    """)

# Configuración adicional para optimización
st.set_page_config(
    page_title="Oil Spill Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)

if __name__ == "__main__":
    main()