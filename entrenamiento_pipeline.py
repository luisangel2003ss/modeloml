import pandas as pd
import numpy as np
import re
import joblib 
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

# --- Función para limpiar números en texto ---
def extraer_numero(texto):
    if pd.isna(texto):
        return np.nan
    match = re.search(r"[-+]?\d*\.\d+|\d+", str(texto))
    return float(match.group()) if match else np.nan

# --- Cargar y preparar dataset ---
df = pd.read_csv("district_cleaned_ready_v2.csv")
df["release_crude_oil"] = df["release_crude_oil"].apply(extraer_numero)
df["release_prod_wtr"] = df["release_prod_wtr"].apply(extraer_numero)
df = df.dropna(subset=["release_crude_oil", "release_prod_wtr"])

X = df.drop(columns=["release_crude_oil", "release_prod_wtr"])
X["recovery_crude_oil_edit"] = df["recovery_crude_oil_edit"]
X["recovery_prod_water_edit"] = df["recovery_prod_water_edit"]

y = df[["release_crude_oil", "release_prod_wtr"]].astype(float)
y_log = np.log1p(y)

# --- Separar datos ---
X_train, X_test, y_train, y_test = train_test_split(X, y_log, test_size=0.2, random_state=42)

# --- Crear preprocesador ---
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

# --- Transformar datos ---
X_train_proc = preprocessor.fit_transform(X_train)
X_test_proc = preprocessor.transform(X_test)

if hasattr(X_train_proc, "toarray"):
    X_train_proc = X_train_proc.toarray()
    X_test_proc = X_test_proc.toarray()

# --- Crear modelo ---
def crear_modelo(input_dim):
    modelo = Sequential([
        Dense(256, activation="relu", input_shape=(input_dim,)),
        Dropout(0.3),
        Dense(128, activation="relu"),
        Dropout(0.2),
        Dense(64, activation="relu"),
        Dense(2)
    ])
    modelo.compile(optimizer=Adam(learning_rate=1e-4), loss="mse", metrics=["mae"])
    return modelo

modelo = crear_modelo(X_train_proc.shape[1])

# --- Entrenamiento ---
early_stop = EarlyStopping(patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-6, verbose=1)

modelo.fit(
    X_train_proc, y_train,
    validation_split=0.2,
    epochs=250,
    batch_size=32,
    callbacks=[early_stop, reduce_lr],
    verbose=2
)

# --- Evaluación ---
y_pred_log = modelo.predict(X_test_proc)
y_pred = np.expm1(y_pred_log)
y_real = np.expm1(y_test)

mse = mean_squared_error(y_real, y_pred, multioutput='raw_values')
mae = mean_absolute_error(y_real, y_pred, multioutput='raw_values')
r2 = r2_score(y_real, y_pred, multioutput='raw_values')

print("\n--- Métricas de Validación ---")
print(f"Crudo -> MSE: {mse[0]:.2f} | MAE: {mae[0]:.2f} | R2: {r2[0]:.2f}")
print(f"Agua  -> MSE: {mse[1]:.2f} | MAE: {mae[1]:.2f} | R2: {r2[1]:.2f}")

# --- Guardar modelo y preprocesador ---
modelo.save("modelo_multisalida.h5")
joblib.dump(preprocessor, "preprocessor.pkl")

# --- Función de predicción para nuevos datos ---
def predecir(nuevos_datos_df):
    df_input = nuevos_datos_df.copy()
    df_input["recovery_crude_oil_edit"] = df_input.get("recovery_crude_oil_edit", np.nan)
    df_input["recovery_prod_water_edit"] = df_input.get("recovery_prod_water_edit", np.nan)

    X_nuevo_proc = preprocessor.transform(df_input)
    if hasattr(X_nuevo_proc, "toarray"):
        X_nuevo_proc = X_nuevo_proc.toarray()

    pred_log = modelo.predict(X_nuevo_proc)
    pred = np.expm1(pred_log)
    return pd.DataFrame(pred, columns=["pred_crudo", "pred_agua"])

# --- Ejemplo de uso de predicción ---
# Puedes reemplazar esto con cualquier otro DataFrame válido con columnas compatibles
ejemplo = X_test.head(3)
predicciones = predecir(ejemplo)
print("\n--- Predicciones para nuevos datos ---")
print(predicciones)
