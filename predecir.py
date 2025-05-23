import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model

def cargar_modelo_y_preprocesador(preproc_path="preprocessor.pkl", modelo_path="modelo_entrenado.h5"):
    preprocessor = joblib.load(preproc_path)
    modelo = load_model(modelo_path)
    return preprocessor, modelo

def validar_columnas(df, columnas_esperadas):
    # Verifica que el DataFrame tenga las columnas necesarias para el modelo
    faltantes = [col for col in columnas_esperadas if col not in df.columns]
    extras = [col for col in df.columns if col not in columnas_esperadas]
    if faltantes:
        raise ValueError(f"Faltan columnas en el dataset: {faltantes}")
    # Opcionalmente, puedes eliminar columnas extras para evitar problemas
    if extras:
        print(f"Se eliminarán columnas extras no usadas: {extras}")
        df = df.drop(columns=extras)
    # Asegurar el orden correcto
    df = df[columnas_esperadas]
    return df

def predecir(df_nuevo, preprocessor, modelo, columnas_esperadas):
    # Validar columnas
    df_nuevo = validar_columnas(df_nuevo, columnas_esperadas)
    
    # Transformar con el preprocesador
    X_proc = preprocessor.transform(df_nuevo)
    
    # Convertir a array si es sparse
    if hasattr(X_proc, "toarray"):
        X_proc = X_proc.toarray()
    
    # Hacer predicción
    preds = modelo.predict(X_proc)
    return preds

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Uso: python predecir.py archivo_entrada.csv")
        sys.exit(1)
    
    archivo_entrada = sys.argv[1]
    
    # Cargar modelo y preprocesador
    preprocessor, modelo = cargar_modelo_y_preprocesador()
    
    # Leer el CSV nuevo
    df_nuevo = pd.read_csv(archivo_entrada)
    
    # Aquí debes definir explícitamente las columnas usadas en entrenamiento (X)
    columnas_usadas = preprocessor.transformers_[0][2] + preprocessor.transformers_[1][2]  # columnas num + cat
    
    try:
        predicciones = predecir(df_nuevo, preprocessor, modelo, columnas_usadas)
        print("Predicciones:")
        print(predicciones)
    except Exception as e:
        print(f"Error en la predicción: {e}")
