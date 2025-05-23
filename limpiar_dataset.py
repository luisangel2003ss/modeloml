# -*- coding: utf-8 -*-
"""
Created on Thu May 22 19:20:54 2025

@author: RAUL SANCHEZ
"""

import pandas as pd

# Archivo original con todas las columnas
archivo_original = "district_cleaned_ready_v2.csv"

# Leer el CSV original
df = pd.read_csv(archivo_original)

# Columnas que no son variables de entrada (targets)
columnas_salida = ["release_crude_oil", "release_prod_wtr"]

# Filtrar columnas necesarias para predecir (features)
columnas_necesarias = [col for col in df.columns if col not in columnas_salida]

# Crear nuevo DataFrame solo con las columnas necesarias
df_limpio = df[columnas_necesarias]

# Guardar archivo limpio
archivo_limpio = "district_cleaned_ready_v2_limpio.csv"
df_limpio.to_csv(archivo_limpio, index=False)

print(f"Archivo limpio guardado en '{archivo_limpio}' con {len(columnas_necesarias)} columnas.")
