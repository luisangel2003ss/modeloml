import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import joblib
import pickle

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.sparse import csr_matrix

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

# ==============================
# Cargar y limpiar los datos
# ==============================

archivo = "district_cleaned_ready_v2.csv"
df = pd.read_csv(archivo)

def extraer_numero(texto):
    if pd.isna(texto):
        return np.nan
    match = re.search(r"[-+]?\d*\.\d+|\d+", str(texto))
    return float(match.group()) if match else np.nan

df["release_crude_oil"] = df["release_crude_oil"].apply(extraer_numero)
df["release_prod_wtr"] = df["release_prod_wtr"].apply(extraer_numero)
df = df.dropna(subset=["release_crude_oil", "release_prod_wtr"])

if "recovery_crude_oil_edit" not in df.columns or "recovery_prod_water_edit" not in df.columns:
    raise ValueError("Las variables recovery_crude_oil_edit y recovery_prod_water_edit no están presentes en el dataset.")

X = df.drop(columns=["release_crude_oil", "release_prod_wtr"])
X["recovery_crude_oil_edit"] = df["recovery_crude_oil_edit"]
X["recovery_prod_water_edit"] = df["recovery_prod_water_edit"]
y_crudo = df["release_crude_oil"].astype(float)
y_agua = df["release_prod_wtr"].astype(float)

# ==============================
# Preprocesamiento
# ==============================

columnas_numericas = X.select_dtypes(include=[np.number]).columns.tolist()
columnas_categoricas = X.select_dtypes(include=["object", "category"]).columns.tolist()

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

X_train, X_test, y_crudo_train, y_crudo_test, y_agua_train, y_agua_test = train_test_split(
    X, y_crudo, y_agua, test_size=0.2, random_state=42
)

X_train_proc = preprocessor.fit_transform(X_train)
X_test_proc = preprocessor.transform(X_test)

if isinstance(X_train_proc, csr_matrix):
    X_train_proc = X_train_proc.toarray()
if isinstance(X_test_proc, csr_matrix):
    X_test_proc = X_test_proc.toarray()

# ==============================
# Función para crear el modelo
# ==============================

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

# ==============================
# Entrenamiento
# ==============================

early_stop = EarlyStopping(patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)

modelo_crudo = crear_modelo(X_train_proc.shape[1])
hist_crudo = modelo_crudo.fit(
    X_train_proc, y_crudo_train,
    epochs=250,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stop, reduce_lr]
)
modelo_crudo.save("modelo_crudo.h5")

modelo_agua = crear_modelo(X_train_proc.shape[1])
hist_agua = modelo_agua.fit(
    X_train_proc, y_agua_train,
    epochs=250,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stop, reduce_lr]
)
modelo_agua.save("modelo_agua.h5")

# ==============================
# Guardar preprocesador e historiales
# ==============================

joblib.dump(preprocessor, "preprocessor.pkl")
with open("training_history_crudo.pkl", "wb") as f:
    pickle.dump(hist_crudo.history, f)
with open("training_history_agua.pkl", "wb") as f:
    pickle.dump(hist_agua.history, f)

# ==============================
# Evaluación
# ==============================

y_pred_crudo = np.clip(modelo_crudo.predict(X_test_proc), 0, None).flatten()
y_pred_agua = np.clip(modelo_agua.predict(X_test_proc), 0, None).flatten()

print(f"\n[Métricas - Crudo]")
print(f"MSE: {mean_squared_error(y_crudo_test, y_pred_crudo):.2f}")
print(f"MAE: {mean_absolute_error(y_crudo_test, y_pred_crudo):.2f}")
print(f"R²:  {r2_score(y_crudo_test, y_pred_crudo):.2f}")

print(f"\n[Métricas - Agua]")
print(f"MSE: {mean_squared_error(y_agua_test, y_pred_agua):.2f}")
print(f"MAE: {mean_absolute_error(y_agua_test, y_pred_agua):.2f}")
print(f"R²:  {r2_score(y_agua_test, y_pred_agua):.2f}")
