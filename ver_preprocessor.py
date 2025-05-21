# -*- coding: utf-8 -*-
"""
Created on Wed May 21 14:52:33 2025

@author: RAUL SANCHEZ
"""

import joblib

# Cargar el preprocesador
preprocessor = joblib.load("preprocessor.pkl")

# Mostrar información del preprocesador
print("\n📦 Tipo de preprocesador:", type(preprocessor))

# Si es ColumnTransformer (lo más probable)
if hasattr(preprocessor, "transformers"):
    print("\n🔍 Estructura del ColumnTransformer:")
    for name, transformer, columns in preprocessor.transformers:
        print(f"\n🧱 Nombre: {name}")
        print(f"📌 Columnas: {columns}")
        print(f"⚙️  Objeto transformer: {transformer}")
        