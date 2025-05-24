import pandas as pd
import joblib
import numpy as np

def cargar_preprocesador():
    preprocessor = joblib.load("preprocessor.pkl")
    return preprocessor

def mostrar_info_preprocesador(preprocessor):
    print("=== Información del preprocesador ===")
    for i, (nombre, trans, columnas) in enumerate(preprocessor.transformers_):
        print(f"Transformador #{i}:")
        print(f"  Nombre: {nombre}")
        print(f"  Columnas usadas: {columnas}")
        if nombre == 'cat':
            for col, encoder in zip(columnas, trans.named_steps['encoder'].categories_):
                print(f"    Categorías para columna '{col}': {len(encoder)} categorías")
    print("-----------------------------------")

def limpiar_columnas_numericas(df, columnas):
    for col in columnas:
        df[col] = df[col].astype(str).str.extract(r'(\d+\.?\d*)')
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

def verificar_compatibilidad(preprocessor, df_nuevo):
    print("Transformando datos de prueba...")
    try:
        X_proc = preprocessor.transform(df_nuevo)
        if hasattr(X_proc, "toarray"):
            X_proc = X_proc.toarray()
        print(f"✅ Transformación exitosa. Forma de los datos procesados: {X_proc.shape}")
    except Exception as e:
        print("❌ Error al transformar los datos:")
        print(e)

def main():
    preprocessor = cargar_preprocesador()
    mostrar_info_preprocesador(preprocessor)

    archivo_prueba = input("Ingresa el path al archivo CSV de prueba con las columnas esperadas: ")
    df_prueba = pd.read_csv(archivo_prueba)

    columnas_usadas = preprocessor.transformers_[0][2] + preprocessor.transformers_[1][2]

    # Eliminar columnas no usadas
    extras = [col for col in df_prueba.columns if col not in columnas_usadas]
    if extras:
        print(f"⚠ Se eliminarán columnas extras no usadas: {extras}")
        df_prueba = df_prueba.drop(columns=extras)

    # Limpiar columnas numéricas
    numeric_features = preprocessor.transformers_[0][2]
    df_prueba = limpiar_columnas_numericas(df_prueba, numeric_features)

    # Asegurar que las columnas estén en el mismo orden
    df_prueba = df_prueba[columnas_usadas]

    verificar_compatibilidad(preprocessor, df_prueba)

if __name__ == "__main__":
    main()
