import pandas as pd
import numpy as np
import re
import joblib
import pickle

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

# --- Función para limpiar números en strings ---
def extraer_numero(texto):
    if pd.isna(texto):
        return np.nan
    import re
    match = re.search(r"[-+]?\d*\.\d+|\d+", str(texto))
    return float(match.group()) if match else np.nan

# --- Cargar y limpiar dataset ---
df = pd.read_csv("district_cleaned_ready_v2.csv")

df["release_crude_oil"] = df["release_crude_oil"].apply(extraer_numero)
df["release_prod_wtr"] = df["release_prod_wtr"].apply(extraer_numero)
df = df.dropna(subset=["release_crude_oil", "release_prod_wtr"])

X = df.drop(columns=["release_crude_oil", "release_prod_wtr"])
X["recovery_crude_oil_edit"] = df["recovery_crude_oil_edit"]
X["recovery_prod_water_edit"] = df["recovery_prod_water_edit"]

y_crudo = df["release_crude_oil"].astype(float)
y_agua = df["release_prod_wtr"].astype(float)

# --- Columnas numéricas y categóricas ---
columnas_numericas = X.select_dtypes(include=[np.number]).columns.tolist()
columnas_categoricas = X.select_dtypes(include=["object", "category"]).columns.tolist()

# --- Crear preprocesador ---
preprocessor = ColumnTransformer([
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]), columnas_numericas),
    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ]), columnas_categoricas)
])

# --- Dividir datos ---
X_train, X_test, y_crudo_train, y_crudo_test, y_agua_train, y_agua_test = train_test_split(
    X, y_crudo, y_agua, test_size=0.2, random_state=42
)

# --- Ajustar y transformar preprocesador ---
X_train_proc = preprocessor.fit_transform(X_train)
X_test_proc = preprocessor.transform(X_test)

# Convertir sparse matrices a arrays si es necesario
if hasattr(X_train_proc, "toarray"):
    X_train_proc = X_train_proc.toarray()
if hasattr(X_test_proc, "toarray"):
    X_test_proc = X_test_proc.toarray()

input_dim = X_train_proc.shape[1]
print(f"Dimensión de entrada para el modelo: {input_dim}")

# --- Crear función para modelo ---
def crear_modelo(input_dim):
    modelo = Sequential([
        Dense(256, activation="relu", input_shape=(input_dim,)),
        Dropout(0.2),
        Dense(128, activation="relu"),
        Dropout(0.2),
        Dense(64, activation="relu"),
        Dense(1)
    ])
    modelo.compile(optimizer=Adam(learning_rate=0.0001), loss="mse", metrics=["mae"])
    return modelo

# --- Callbacks ---
early_stop = EarlyStopping(patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)

# --- Entrenar modelo para crudo ---
modelo_crudo = crear_modelo(input_dim)
modelo_crudo.fit(
    X_train_proc, y_crudo_train,
    epochs=250,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stop, reduce_lr],
    verbose=2
)
modelo_crudo.save("modelo_crudo.h5")

# --- Entrenar modelo para agua ---
modelo_agua = crear_modelo(input_dim)
modelo_agua.fit(
    X_train_proc, y_agua_train,
    epochs=250,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stop, reduce_lr],
    verbose=2
)
modelo_agua.save("modelo_agua.h5")

# --- Guardar preprocesador ---
joblib.dump(preprocessor, "preprocessor.pkl")
print("Preprocesador guardado.")

# --- Evaluación ---
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

y_pred_crudo = np.clip(modelo_crudo.predict(X_test_proc), 0, None).flatten()
y_pred_agua = np.clip(modelo_agua.predict(X_test_proc), 0, None).flatten()

print(f"\nMétricas Crudo:")
print(f"MSE: {mean_squared_error(y_crudo_test, y_pred_crudo):.2f}")
print(f"MAE: {mean_absolute_error(y_crudo_test, y_pred_crudo):.2f}")
print(f"R2:  {r2_score(y_crudo_test, y_pred_crudo):.2f}")

print(f"\nMétricas Agua:")
print(f"MSE: {mean_squared_error(y_agua_test, y_pred_agua):.2f}")
print(f"MAE: {mean_absolute_error(y_agua_test, y_pred_agua):.2f}")
print(f"R2:  {r2_score(y_agua_test, y_pred_agua):.2f}")
