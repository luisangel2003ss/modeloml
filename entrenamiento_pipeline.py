import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import joblib
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping

# Cargar el dataset
ruta = "district_cleaned_ready_v2.csv"  # Reemplaza con la ruta correcta si es necesario
df = pd.read_csv(ruta)

# Eliminar columnas no útiles (opcional, ajusta según necesidad)
df = df.drop(columns=["duplicate"], errors="ignore")

# Definir columnas numéricas y categóricas relevantes
columnas_numericas = [
    "release_crude_oil", "release_cond", "release_prod_wtr", "release_gas",
    "recovery_crude_oil", "recovery_cond", "recovery_prod_wtr"
]
columnas_categoricas = [
    "type_operation", "probable_cause", "county", "district_edit"
]

# Función para limpiar valores como "5 GAL", "10 BBL", etc.
def limpiar_valores(columna):
    return columna.astype(str).str.extract(r'(\d+\.?\d*)')[0].astype(float)

# Aplicar limpieza a las columnas numéricas que contienen texto
for col in columnas_numericas:
    if col in df.columns:
        df[col] = limpiar_valores(df[col])

# Eliminar filas sin la variable objetivo
objetivo = "release_crude_oil"
df = df[df[objetivo].notna()]

# Definir X e y
X = df[columnas_numericas + columnas_categoricas].copy()
y = df[[objetivo, "release_prod_wtr"]].copy()

# Convertir y a valores numéricos (solo por si acaso)
y[objetivo] = pd.to_numeric(y[objetivo], errors='coerce')
y["release_prod_wtr"] = pd.to_numeric(y["release_prod_wtr"], errors='coerce')

# Separar en train y test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocesamiento
preprocessor = ColumnTransformer(
    transformers=[
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]), columnas_numericas),

        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ]), columnas_categoricas)
    ]
)

# Ajustar el preprocesador
preprocessor.fit(X_train)

# Transformar los datos
X_train_proc = preprocessor.fit_transform(X_train)
# Convertir sparse matrix a dense array (necesario para Keras)
if hasattr(X_train_proc, "toarray"):
    X_train_proc = X_train_proc.toarray()
X_test_proc = preprocessor.fit_transform(X_test)
# Convertir sparse matrix a dense array (necesario para Keras)
if hasattr(X_test_proc, "toarray"):
    X_test_proc = X_test_proc.toarray()


# Construir el modelo
modelo = Sequential([
    Dense(64, activation='relu', input_shape=(X_train_proc.shape[1],)),
    Dense(32, activation='relu'),
    Dense(2)  # Dos salidas: release_crude_oil y release_prod_wtr
])

modelo.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Entrenar el modelo
modelo.fit(
    X_train_proc, y_train,
    validation_split=0.2,
    epochs=100,
    callbacks=[EarlyStopping(patience=10, restore_best_weights=True)],
    verbose=1
)

# Guardar preprocesador y modelo
joblib.dump(preprocessor, "preprocessor.pkl")
modelo.save("modelo_entrenado.h5")

print("Entrenamiento y guardado completados correctamente.")
