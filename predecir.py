import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import os

def cargar_modelo_y_preprocesador(preproc_path="preprocessor.pkl", modelo_path="modelo_entrenado.h5"):
    preprocessor = joblib.load(preproc_path)
    modelo = load_model(modelo_path)
    return preprocessor, modelo

def extraer_columnas_usadas(preprocessor):
    columnas_numericas = preprocessor.transformers_[0][2]
    columnas_categoricas = preprocessor.transformers_[1][2]
    return list(columnas_numericas) + list(columnas_categoricas)

def validar_y_limpiar_columnas(df, columnas_esperadas):
    faltantes = [col for col in columnas_esperadas if col not in df.columns]
    if faltantes:
        raise ValueError(f"Faltan columnas en el dataset: {faltantes}")
    extras = [col for col in df.columns if col not in columnas_esperadas]
    if extras:
        print(f"Se eliminarán columnas extras no usadas: {extras}")
        df = df.drop(columns=extras)
    df = df[columnas_esperadas]  # Reordenar columnas
    return df

def guardar_dataset_limpio(df, archivo_salida="district_predicciones_limpio.csv"):
    df.to_csv(archivo_salida, index=False)
    print(f"Dataset limpio guardado en '{archivo_salida}'")
    return archivo_salida

def predecir(df_nuevo, preprocessor, modelo):
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
    columnas_usadas = extraer_columnas_usadas(preprocessor)

    df_nuevo = pd.read_csv(archivo_entrada)

    try:
        df_limpio = validar_y_limpiar_columnas(df_nuevo, columnas_usadas)
        archivo_limpio = guardar_dataset_limpio(df_limpio)
        predicciones = predecir(df_limpio, preprocessor, modelo)
        print("Predicciones:")
        for i, fila in enumerate(predicciones):
            # Si tu modelo predice crudo y agua en salida separada, ajusta esto:
            if fila.shape == (2,):
                crude, water = fila
                print(f"Fila {i+1} → Crudo: {crude:.2f} bbl, Agua: {water:.2f} bbl")
            else:
                # Si solo predice uno, muestra solo ese valor
                print(f"Fila {i+1} → Predicción: {fila[0]:.2f}")
    except Exception as e:
        print(f"Error en la predicción: {e}")
