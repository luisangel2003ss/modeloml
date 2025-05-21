# Versión final corregida
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
        if df[col].notna().any():  # Solo si hay valores numéricos
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
        learning_rate=0.0003
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

def main():
    X, y = load_data()
    
    preprocessor = build_preprocessor()
    X_processed = preprocessor.fit_transform(X)
    
    # Estratificación basada en y (convertido a DataFrame temporal)
    y_df = pd.DataFrame(y)
    y_mean = y_df.mean(axis=1)
    bins = np.quantile(y_mean, np.linspace(0, 1, 5))
    y_binned = np.digitize(y_mean, bins)
    
    # División con estratificación (convertimos a array después)
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y, test_size=0.2, 
        stratify=y_binned, random_state=42
    )
    
    # Para la validación, necesitamos nuevos bins basados en y_train
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

if __name__ == "__main__":
    main()