import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# Cargar datos limpios sin columnas target
archivo = "district_cleaned_ready_v2_limpio.csv"
df = pd.read_csv(archivo)

# Cargar preprocesador y modelo
preprocessor = joblib.load("preprocessor.pkl")
modelo = load_model("modelo_entrenado.h5")

# Transformar los datos con el preprocesador
X_proc = preprocessor.transform(df)

# Convertir a array denso si es sparse
from scipy.sparse import csr_matrix
if isinstance(X_proc, csr_matrix):
    X_proc = X_proc.toarray()

# Predecir
predicciones = modelo.predict(X_proc)

# Agregar predicciones al DataFrame original (opcional)
df["pred_release_crude_oil"] = predicciones[:, 0]
df["pred_release_prod_wtr"] = predicciones[:, 1]

# Guardar resultado con predicciones
df.to_csv("district_predicciones.csv", index=False)

print("Predicción completada y guardada en district_predicciones.csv")
