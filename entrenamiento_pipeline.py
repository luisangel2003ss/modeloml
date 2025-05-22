import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

# 1. Carga datos (ajusta ruta)
ruta_csv = "district_cleaned_ready_v2.csv"
df = pd.read_csv(ruta_csv)

# 2. Define X, y y limpia columnas objetivo
X = df.drop(columns=["release_crude_oil", "release_prod_wtr"])
y = df[["release_crude_oil", "release_prod_wtr"]]

y["release_crude_oil"] = pd.to_numeric(y["release_crude_oil"], errors='coerce')
y["release_prod_wtr"] = pd.to_numeric(y["release_prod_wtr"], errors='coerce')

# Eliminar filas con NaN en y (para evitar problemas)
mask = y.notnull().all(axis=1)
X = X[mask]
y = y[mask]

# Transformamos objetivos (log1p)
y = np.log1p(y)

# 3. Define columnas numéricas y categóricas (ajusta según tus columnas)
numeric_features = ["release_cond", "release_gas"]
categorical_features = ["probable_cause_edit", "type_operation"]

numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', RobustScaler())
])

categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer([
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])

# 4. Divide datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Ajusta preprocesador con X_train
preprocessor.fit(X_train)

# 6. Transforma X_train para obtener input_dim para modelo Keras
X_train_proc = preprocessor.transform(X_train)
input_dim = X_train_proc.shape[1]

# 7. Construye modelo Keras simple
def construir_modelo(input_dim):
    model = Sequential([
        Dense(64, activation='relu', input_shape=(input_dim,)),
        Dense(32, activation='relu'),
        Dense(2)  # dos salidas: crude oil y prod water
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

modelo_keras = construir_modelo(input_dim)

# 8. Entrena modelo Keras con datos preprocesados
modelo_keras.fit(X_train_proc, y_train, epochs=50, batch_size=32, verbose=1)

# 9. Guarda preprocesador con joblib
joblib.dump(preprocessor, "preprocessor.pkl")

# 10. Guarda modelo Keras con su método nativo
modelo_keras.save("modelo_keras.h5")

print("Preprocesador guardado en 'preprocessor.pkl'")
print("Modelo Keras guardado en 'modelo_keras.h5'")
