# -*- coding: utf-8 -*-
"""
Created on Mon May 19 22:50:38 2025

@author: RAUL SANCHEZ
"""

import pandas as pd
import numpy as np
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
    
    # Selección de características
    features = ['release_crude_oil_edit', 'release_prod_water_edit', 
                'release_cond', 'release_gas', 'probable_cause_edit', 
                'type_operation', 'county_edit', 'operator_edit']
    
    df = df[features].dropna(subset=features)
    
    # Conversión numérica segura
    numeric_cols = ['release_cond', 'release_gas']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        if df[col].notna().any():
            upper_limit = df[col].quantile(0.99)
            df[col] = np.clip(df[col], 0, upper_limit)
    
    # Targets numéricos
    targets = ['release_crude_oil_edit', 'release_prod_water_edit']
    for target_col in targets:
        df[target_col] = pd.to_numeric(df[target_col], errors='coerce')
        if df[target_col].notna().any():
            upper_limit = df[target_col].quantile(0.98)
            df = df[df[target_col] <= upper_limit]
    
    X = df.drop(targets, axis=1)
    y = np.log1p(df[targets].values)
    
    return X, y

def build_preprocessor():
    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', RobustScaler())
    ])
    
    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('encoder', OneHotEncoder(
            handle_unknown='ignore', 
            sparse_output=False,
            min_frequency=0.02))
    ])
    
    return ColumnTransformer([
        ('num', numeric_transformer, ['release_cond', 'release_gas']),
        ('cat', categorical_transformer, 
         ['probable_cause_edit', 'type_operation', 'county_edit', 'operator_edit'])
    ])

def build_stable_model(input_shape):
    model = tf.keras.Sequential([
        layers.Dense(128, activation='relu', 
                    kernel_initializer='he_normal',
                    kernel_regularizer=regularizers.l1_l2(0.001, 0.01)),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        layers.Dense(64, activation='relu',
                    kernel_regularizer=regularizers.l2(0.005)),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(32, activation='relu'),
        layers.Dense(2)
    ])
    
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=0.0002
    )
    
    model.compile(
        optimizer=optimizer,
        loss='huber_loss',
        metrics=['mae', 'mse']
    )
    return model

def train_model(model, X_train, y_train, X_val, y_val):
    early_stop = callbacks.EarlyStopping(
        monitor='val_mse',
        patience=30,
        restore_best_weights=True,
        min_delta=0.001
    )
    
    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor='val_mse',
        factor=0.3,
        patience=15,
        min_lr=1e-6,
        verbose=1
    )
    
    history = model.fit(
        X_train,
        y_train,
        epochs=400,
        batch_size=75,
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
    print("Ingrese los valores para las siguientes características:")
    
    # Solicitar inputs
    release_cond = float(input("release_cond: "))
    release_gas = float(input("release_gas: "))
    probable_cause_edit = input("probable_cause_edit: ")
    type_operation = input("type_operation: ")
    county_edit = input("county_edit: ")
    operator_edit = input("operator_edit: ")
    
    # Crear DataFrame con los inputs
    new_data = pd.DataFrame({
        'release_cond': [release_cond],
        'release_gas': [release_gas],
        'probable_cause_edit': [probable_cause_edit],
        'type_operation': [type_operation],
        'county_edit': [county_edit],
        'operator_edit': [operator_edit]
    })
    
    # Preprocesar los datos
    processed_data = preprocessor.transform(new_data)
    
    # Realizar predicción
    prediction = model.predict(processed_data)
    prediction = np.expm1(prediction)  # Revertir transformación logarítmica
    
    print("\nPredicción:")
    print(f"Crudo derramado: {prediction[0][0]:.2f}")
    print(f"Agua de producción derramada: {prediction[0][1]:.2f}")
    
    return prediction

def main():
    X, y = load_data()
    
    preprocessor = build_preprocessor()
    X_processed = preprocessor.fit_transform(X)
    
    # Estratificación basada en y
    y_df = pd.DataFrame(y)
    y_mean = y_df.mean(axis=1)
    bins = np.quantile(y_mean, np.linspace(0, 1, 5))
    y_binned = np.digitize(y_mean, bins)
    
    # División de datos
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y, test_size=0.2, 
        stratify=y_binned, random_state=42
    )
    
    # Estratificación para validación
    y_train_mean = pd.DataFrame(y_train).mean(axis=1)
    train_bins = np.quantile(y_train_mean, np.linspace(0, 1, 5))
    y_train_binned = np.digitize(y_train_mean, train_bins)
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2,
        stratify=y_train_binned, random_state=42
    )
    
    model = build_stable_model(X_train.shape[1])
    history = train_model(model, X_train, y_train, X_val, y_val)
    
    plot_mse(history)
    
    # Evaluación
    y_pred = np.expm1(model.predict(X_test))
    y_test_orig = np.expm1(y_test)
    
    print("Métricas finales:")
    for i, name in enumerate(['Crudo', 'Agua']):
        print(f"\n{name}:")
        print(f"MSE: {mean_squared_error(y_test_orig[:,i], y_pred[:,i]):.2f}")
        print(f"R²: {r2_score(y_test_orig[:,i], y_pred[:,i]):.2f}")
    
    # Predicción interactiva
    while True:
        try:
            predict_new_samples(model, preprocessor)
            another = input("\n¿Desea hacer otra predicción? (s/n): ")
            if another.lower() != 's':
                break
        except Exception as e:
            print(f"Error: {e}. Intente nuevamente.")

if __name__ == "__main__":
    main()