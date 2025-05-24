import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model

def cargar_modelo_y_preprocesador(preproc_path="preprocessor.pkl", modelo_path="modelo_entrenado.h5"):
    preprocessor = joblib.load(preproc_path)
    modelo = load_model(modelo_path)
    return preprocessor, modelo

def validar_columnas(df, columnas_esperadas):
    faltantes = [col for col in columnas_esperadas if col not in df.columns]
    extras = [col for col in df.columns if col not in columnas_esperadas]
    if faltantes:
        raise ValueError(f"Faltan columnas en el dataset: {faltantes}")
    if extras:
        print(f"Se eliminarán columnas extras no usadas: {extras}")
        df = df.drop(columns=extras)
    df = df[columnas_esperadas]
    return df

def predecir(df_nuevo, preprocessor, modelo, columnas_esperadas):
    df_nuevo = validar_columnas(df_nuevo, columnas_esperadas)
    X_proc = preprocessor.transform(df_nuevo)
    if hasattr(X_proc, "toarray"):
        X_proc = X_proc.toarray()
    preds = modelo.predict(X_proc)
    preds = np.clip(preds, 0, None)  # evitar negativos
    return preds

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Uso: python predecir.py archivo_entrada.csv")
        sys.exit(1)
    
    archivo_entrada = sys.argv[1]
    
    preprocessor, modelo = cargar_modelo_y_preprocesador()
    
    df_nuevo = pd.read_csv(archivo_entrada)

    # Extraer columnas usadas del preprocesador
    columnas_numericas = preprocessor.transformers_[0][2]
    columnas_categoricas = preprocessor.transformers_[1][2]
    columnas_usadas = columnas_numericas + columnas_categoricas

    try:
        predicciones = predecir(df_nuevo, preprocessor, modelo, columnas_usadas)
        print("Predicciones:")
        for i, fila in enumerate(predicciones):
            crude, water = fila
            print(f"Fila {i+1} → Crudo: {crude:.2f} bbl, Agua: {water:.2f} bbl")
    except Exception as e:
        print(f"Error en la predicción: {e}")
