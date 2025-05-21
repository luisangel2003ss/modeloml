import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras import layers, regularizers, callbacks
import matplotlib.pyplot as plt

def load_data():
    df = pd.read_csv('district_cleaned_ready_v2.csv')

    # Selección de variables esenciales
    features = ['release_crude_oil_edit', 'release_prod_water_edit',
                'release_cond', 'release_gas',
                'probable_cause_edit', 'type_operation']

    df = df[features].dropna()

    # Conversión a numérico y recorte de outliers
    numeric_cols = ['release_cond', 'release_gas']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        if df[col].notna().any():
            upper_limit = df[col].quantile(0.99)
            df[col] = np.clip(df[col], 0, upper_limit)

    # Recorte de valores extremos para los targets
    for target_col in ['release_crude_oil_edit', 'release_prod_water_edit']:
        df[target_col] = pd.to_numeric(df[target_col], errors='coerce')
        if df[target_col].notna().any():
            upper_limit = df[target_col].quantile(0.98)
            df = df[df[target_col] <= upper_limit]

    X = df.drop(['release_crude_oil_edit', 'release_prod_water_edit'], axis=1)
    y = np.log1p(df[['release_crude_oil_edit', 'release_prod_water_edit']].values)

    return X, y

def build_preprocessor():
    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', RobustScaler())
    ])

    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    return ColumnTransformer([
        ('num', numeric_transformer, ['release_cond', 'release_gas']),
        ('cat', categorical_transformer, ['probable_cause_edit', 'type_operation'])
    ])

def build_model(input_shape):
    model = tf.keras.Sequential([
        layers.Dense(128, activation='relu', kernel_initializer='he_normal',
                     kernel_regularizer=regularizers.l1_l2(0.001, 0.01)),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.005)),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(32, activation='relu'),
        layers.Dense(2)
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
        loss='huber_loss',
        metrics=['mae', 'mse']
    )

    return model

def train_model(model, X_train, y_train, X_val, y_val):
    early_stop = callbacks.EarlyStopping(monitor='val_mse', patience=25, restore_best_weights=True)
    reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_mse', factor=0.3, patience=10, min_lr=1e-6)

    history = model.fit(
        X_train, y_train,
        epochs=300,
        batch_size=64,
        validation_data=(X_val, y_val),
        callbacks=[early_stop, reduce_lr],
        verbose=1
    )

    return history

def plot_mse(history):
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['mse'], label='Entrenamiento')
    plt.plot(history.history['val_mse'], label='Validación')
    plt.title('Evolución del MSE')
    plt.xlabel('Épocas')
    plt.ylabel('MSE')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

def predict_new_samples(model, preprocessor):
    print("\n--- Predicción de nuevos ejemplos ---")
    print("Ingrese valores dentro de estos rangos:")
    print("- Condensado Derramado: 0 a 100")
    print("- Gas Liberado: 0 a 5000")

    def get_valid_input(prompt, min_val, max_val):
        while True:
            try:
                value = float(input(prompt))
                if min_val <= value <= max_val:
                    return value
                else:
                    print(f"Valor fuera de rango ({min_val}-{max_val}). Intente otra vez.")
            except ValueError:
                print("Entrada inválida. Ingrese un número válido.")

    release_cond = get_valid_input("Condensado Derramado: ", 0, 100)
    release_gas = get_valid_input("Gas Liberado: ", 0, 5000)

    probable_cause_edit = input("Causa probable (ej: CORROSION, HUMAN ERROR): ").upper()
    type_operation = input("Tipo de operación (ej: PRODUCTION, DRILLING): ").upper()

    new_data = pd.DataFrame({
        'release_cond': [release_cond],
        'release_gas': [release_gas],
        'probable_cause_edit': [probable_cause_edit],
        'type_operation': [type_operation]
    })

    try:
        processed_data = preprocessor.transform(new_data)
        prediction = model.predict(processed_data)
        prediction = np.expm1(prediction)
        prediction = np.maximum(prediction, 0)

        print("\nPredicción estimada (en barriles):")
        print(f"- Crudo derramado: {prediction[0][0]:.2f}")
        print(f"- Agua de producción: {prediction[0][1]:.2f}")
    except Exception as e:
        print(f"Error durante la predicción: {e}")
        print("Verifique las categorías ingresadas. Use valores comunes.")

def main():
    X, y = load_data()
    preprocessor = build_preprocessor()
    X_processed = preprocessor.fit_transform(X)

    y_mean = pd.DataFrame(y).mean(axis=1)
    bins = np.quantile(y_mean, np.linspace(0, 1, 5))
    y_binned = np.digitize(y_mean, bins)

    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y, test_size=0.2, stratify=y_binned, random_state=42)

    y_train_mean = pd.DataFrame(y_train).mean(axis=1)
    train_bins = np.quantile(y_train_mean, np.linspace(0, 1, 5))
    y_train_binned = np.digitize(y_train_mean, train_bins)

    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, stratify=y_train_binned, random_state=42)

    model = build_model(X_train.shape[1])
    history = train_model(model, X_train, y_train, X_val, y_val)

    plot_mse(history)

    model.save('modelo_entrenado.h5')
    joblib.dump(preprocessor, 'preprocessor.pkl')

    y_pred = np.expm1(model.predict(X_test))
    y_test_orig = np.expm1(y_test)

    print("\nMétricas finales:")
    for i, label in enumerate(['Crudo', 'Agua']):
        print(f"\n{label}:")
        print(f"- MSE: {mean_squared_error(y_test_orig[:, i], y_pred[:, i]):.2f}")
        print(f"- R²: {r2_score(y_test_orig[:, i], y_pred[:, i]):.2f}")

    while True:
        predict_new_samples(model, preprocessor)
        if input("\n¿Desea hacer otra predicción? (s/n): ").lower() != 's':
            break

if __name__ == "__main__":
    main()
