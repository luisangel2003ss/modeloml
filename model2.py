# -*- coding: utf-8 -*-
"""
Created on Mon May 19 05:55:39 2025

@author: RAUL SANCHEZ
"""

# importar librerias
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

#leer el dataset
data = pd.read_csv("houses.csv")
print("Datos cargados exitosamente:\n", data.head())

#estadísticas y visualizacion
custom_stats = pd.DataFrame({
    "Promedio": data.mean(),
    "Mediana": data.median(),
    "Máximo": data.max(),
    "Mínimo": data.min(),
    "Desviación Estándar": data.std(),
    "Rango": data.max() - data.min()
    })
print("Estadísticas personalizadas:")
print(custom_stats)

# preparar datos para elmodelo
X = data[["size", "rooms"]]  # Variable independiente
y = data["price"]  # Variable dependiente

# dividir en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)

# crear el modelo
model = LinearRegression()
model.fit(X_train, y_train)

# evaluar el modelo
y_pred = model.predict(X_test)
print("Métricas del modelo:")
print("Error cuadrático medio (MSE):", mean_squared_error(y_test, y_pred))
print("Coeficiente de determinación (R²):", r2_score(y_test, y_pred))

# valores del peso y del bias
print("\nCoeficientes del modelo:")
print("Bias:", model.intercept_)
print("Pesos:", model.coef_)

# nuevos datos
nuevos_datos_df = pd.DataFrame([[150, 3],[200,4]], columns=['size', 'rooms'])

# valor de las predicciosn
predicciones = model.predict(nuevos_datos_df)
predicciones_formateadas = [f"{pred:,.2f}" for pred in predicciones]

print("Predicciones para nuevos datos:", predicciones_formateadas)