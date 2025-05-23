import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
import joblib
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from scipy.sparse import csr_matrix
import re

# Cargar datos desde CSV
archivo = "district_cleaned_ready_v2.csv"
df = pd.read_csv(archivo)

# Función para extraer número de cadenas tipo '+/- 72' o con otros caracteres
def extraer_numero(texto):
    if pd.isna(texto):
        return np.nan
    match = re.search(r"[-+]?\d*\.\d+|\d+", str(texto))
    if match:
        return float(match.group())
    else:
        return np.nan

# Aplicar limpieza a las columnas objetivo
df["release_crude_oil"] = df["release_crude_oil"].apply(extraer_numero)
df["release_prod_wtr"] = df["release_prod_wtr"].apply(extraer_numero)

# Eliminar filas con etiquetas NaN después de limpieza
df = df.dropna(subset=["release_crude_oil", "release_prod_wtr"])

# Separar variables independientes y dependientes
X = df.drop(columns=["release_crude_oil", "release_prod_wtr"])
y = df[["release_crude_oil", "release_prod_wtr"]].astype(float)

# Identificar columnas numéricas y categóricas
columnas_numericas = X.select_dtypes(include=[np.number]).columns.tolist()
columnas_categoricas = X.select_dtypes(include=["object", "category"]).columns.tolist()

# Preprocesador
preprocessor = ColumnTransformer(
    transformers=[
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]), columnas_numericas),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]), columnas_categoricas)
    ]
)

# Separar en train y test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Ajustar el preprocesador y transformar los datos
X_train_proc = preprocessor.fit_transform(X_train)
X_test_proc = preprocessor.transform(X_test)

# Verificar datos válidos
print("¿NaNs en X_train_proc?", np.isnan(X_train_proc.toarray()).any() if isinstance(X_train_proc, csr_matrix) else np.isnan(X_train_proc).any())
print("¿Infs en X_train_proc?", np.isinf(X_train_proc.toarray()).any() if isinstance(X_train_proc, csr_matrix) else np.isinf(X_train_proc).any())
print("¿NaNs en y_train?", np.isnan(y_train).any().any())
print("¿Infs en y_train?", np.isinf(y_train).any().any())

# Convertir X_train_proc a array denso si es necesario
if isinstance(X_train_proc, csr_matrix):
    X_train_proc = X_train_proc.toarray()
if isinstance(X_test_proc, csr_matrix):
    X_test_proc = X_test_proc.toarray()

# Modelo Keras
modelo = Sequential([
    Dense(64, activation="relu", input_shape=(X_train_proc.shape[1],)),
    Dense(32, activation="relu"),
    Dense(2)  # salida para dos valores
])

modelo.compile(optimizer="adam", loss="mse", metrics=["mae"])

# Entrenamiento
early_stop = EarlyStopping(patience=10, restore_best_weights=True)

hist = modelo.fit(
    X_train_proc, y_train.values,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stop]
)

# Guardar historial de entrenamiento para graficar después
import pickle
with open("training_history.pkl", "wb") as f:
    pickle.dump(hist.history, f)


# Guardar preprocesador y modelo
joblib.dump(preprocessor, "preprocessor.pkl")
modelo.save("modelo_entrenado.h5")

print("Entrenamiento y guardado completados correctamente.")
