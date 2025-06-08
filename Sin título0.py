import os
import pandas as pd

# Definir la ruta absoluta del archivo
ruta_archivo = r"C:\Users\RAUL SANCHEZ\ml\TABLA FINAL PROYECTO CVS.csv"

# Verificar si el archivo existe antes de cargarlo
if os.path.exists(ruta_archivo):
    print("El archivo existe, intentando cargarlo...")

    try:
        # Leer el CSV con el delimitador correcto
        df = pd.read_csv(ruta_archivo, delimiter=",", encoding="latin1", on_bad_lines="skip")
        print("Dataset cargado correctamente.")

        # Definir las columnas relevantes
        columnas_relevantes = [
            "Distrito", "Year_date_of_spill_edit", "Mes_date_of_spill_edit",
            "Dia_date_of_spill_edit", "release_prod_wtr", "operator_rp",
            "Tipo_operacion_rrc", "probable_cause", "probable_cause_edit",
            "spill_sustancias", "release_crude_oil", "release_cond",
            "release_gas", "recovery_crude_oil", "recovery_cond",
            "recovery_prod_wtr", "affected_area", "Grado_sensibilidad_area",
            "Sensibilidad_area", "cleanup_method", "Metodos_aplicados"
        ]

        # Asegurar que las columnas están correctamente identificadas
        df.columns = df.columns.str.strip()  # Eliminar espacios en los nombres
        df.columns = df.columns.str.replace("\ufeff", "")  # Quitar caracteres invisibles

        # Filtrar el dataset para conservar solo las columnas relevantes
        df_filtrado = df[columnas_relevantes]

        # Guardar el nuevo dataset limpio en la misma carpeta del archivo original
        ruta_salida = os.path.join(os.path.dirname(ruta_archivo), "dataset_filtrado.csv")
        df_filtrado.to_csv(ruta_salida, index=False)

        print(f"El dataset ha sido limpiado y guardado como '{ruta_salida}'.")

    except KeyError:
        print("Error: Algunas columnas especificadas no existen en el dataset.")
        print("Verifica los nombres de las columnas con 'print(df.columns)'.")

    except Exception as e:
        print(f"Error al leer el CSV: {e}")

else:
    print("Error: El archivo no se encuentra en la ruta especificada.")