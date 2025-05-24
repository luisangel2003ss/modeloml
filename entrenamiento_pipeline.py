    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    import joblib
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.optimizers import Adam
    from scipy.sparse import csr_matrix
    import re
    import pickle
    import matplotlib.pyplot as plt

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

# ==============================
# Verificar y añadir las nuevas variables
# ==============================

if "recovery_crude_oil_edit" not in df.columns or "recovery_prod_water_edit" not in df.columns:
    raise ValueError("Las variables recovery_crude_oil_edit y recovery_prod_water_edit no están presentes en el dataset.")

X = df.drop(columns=["release_crude_oil", "release_prod_wtr"])
X["recovery_crude_oil_edit"] = df["recovery_crude_oil_edit"]
X["recovery_prod_water_edit"] = df["recovery_prod_water_edit"]
y = df[["release_crude_oil", "release_prod_wtr"]].astype(float)

# ==============================
# Actualizar el pipeline de preprocesamiento
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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train_proc = preprocessor.fit_transform(X_train)
X_test_proc = preprocessor.transform(X_test)

if isinstance(X_train_proc, csr_matrix):
    X_train_proc = X_train_proc.toarray()
if isinstance(X_test_proc, csr_matrix):
    X_test_proc = X_test_proc.toarray()

# ==============================
# Modelo con Dropout
# ==============================

modelo = Sequential([
    Dense(256, activation="relu", input_shape=(X_train_proc.shape[1],)),
    Dropout(0.2),
    Dense(128, activation="relu"),
    Dropout(0.2),
    Dense(64, activation="relu"),
    Dense(2)
])

opt = Adam(learning_rate=0.0001)
modelo.compile(optimizer=opt, loss="mse", metrics=["mae"])

early_stop = EarlyStopping(patience=20, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6, verbose=1)

# ==============================
# Entrenamiento
# ==============================

hist = modelo.fit(
    X_train_proc, y_train.values,
    epochs=300,
    batch_size=64,
    validation_split=0.2,
    callbacks=[early_stop, reduce_lr]
)

# Guardar el historial y modelo
with open("training_history.pkl", "wb") as f:
    pickle.dump(hist.history, f)

joblib.dump(preprocessor, "preprocessor.pkl")
modelo.save("modelo_entrenado.h5")
print("Entrenamiento y guardado completados correctamente.")

# ==============================
# Predicciones y métricas
# ==============================

y_pred = modelo.predict(X_test_proc)
y_pred = np.clip(y_pred, 0, None)  # Limitar a positivos

mse_crude = mean_squared_error(y_test["release_crude_oil"], y_pred[:, 0])
mae_crude = mean_absolute_error(y_test["release_crude_oil"], y_pred[:, 0])
r2_crude = r2_score(y_test["release_crude_oil"], y_pred[:, 0])

mse_water = mean_squared_error(y_test["release_prod_wtr"], y_pred[:, 1])
mae_water = mean_absolute_error(y_test["release_prod_wtr"], y_pred[:, 1])
r2_water = r2_score(y_test["release_prod_wtr"], y_pred[:, 1])

print(f"\n[Métricas - Crude Oil]")
print(f"MSE: {mse_crude:.2f}, MAE: {mae_crude:.2f}, R²: {r2_crude:.2f}")

print(f"\n[Métricas - Produced Water]")
print(f"MSE: {mse_water:.2f}, MAE: {mae_water:.2f}, R²: {r2_water:.2f}")

# ==============================
# Gráfica mejorada de la pérdida
# ==============================

def suavizar(lista, ventana=5):
    return pd.Series(lista).rolling(ventana, min_periods=1, center=True).mean()

plt.figure(figsize=(10, 6))
plt.plot(suavizar(hist.history['loss']), label='Entrenamiento MSE', color='royalblue', linewidth=2)
plt.plot(suavizar(hist.history['val_loss']), label='Validación MSE', color='darkorange', linewidth=2)
plt.title('Curva de pérdida (MSE) durante el entrenamiento', fontsize=14)
plt.xlabel('Épocas')
plt.ylabel('Error cuadrático medio (MSE)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# ==============================
# Gráfica de predicciones vs reales
# ==============================

fig, axs = plt.subplots(1, 2, figsize=(12, 5))

axs[0].scatter(y_test["release_crude_oil"], y_pred[:, 0], alpha=0.4, color='teal')
axs[0].plot([0, max(y_test["release_crude_oil"])], [0, max(y_test["release_crude_oil"])], 'r--')
axs[0].set_title(f"Crude Oil\nR²={r2_crude:.2f}, MAE={mae_crude:.1f}")
axs[0].set_xlabel("Real")
axs[0].set_ylabel("Predicción")

axs[1].scatter(y_test["release_prod_wtr"], y_pred[:, 1], alpha=0.4, color='purple')
axs[1].plot([0, max(y_test["release_prod_wtr"])], [0, max(y_test["release_prod_wtr"])], 'r--')
axs[1].set_title(f"Produced Water\nR²={r2_water:.2f}, MAE={mae_water:.1f}")
axs[1].set_xlabel("Real")
axs[1].set_ylabel("Predicción")

plt.tight_layout()
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()
