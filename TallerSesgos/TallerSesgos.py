#Importación de librerías
import pandas as pd #Manipulación y análisis de datos en tablas (DataFrames)
import numpy as np #Operaciones númericas y manejo de valores nulos
from pathlib import Path #Gestión rutas de archivos
import matplotlib.pyplot as plt #Visualización
import seaborn as sns #Gráficos estadísticos

pd.set_option("display.max_columns", None) #Mostrar todas las columnas sin recortar con puntos suspensivos (...)
sns.set_style("whitegrid") #visual con fondo blanco y cuadrícula gris clara a todos los gráficos

# 1. Configuración de Rutas del Entorno
SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent.parent

FILE_ADULT = "bank-full.csv" 



#if not FILE_ADULT.exists():
#    print(f"El archivo no existe en la ruta: {FILE_ADULT}")
#else:
#    print("¡Archivo encontrado correctamente!")

df = pd.read_csv(FILE_ADULT, sep=";") #Función que lee el CSV 
#y lo convierte en un DataFrame (df).
#sep: separador del punto y coma ;

print(df.head())

################## PASO 1. EXPLORACIÓN INICIAL (EDA)

print("\nEstructura:", df.shape) #Devuelve una tupla con dos números con la estructura (número_de_filas, número_de_columnas)
print("\n##################Información General##############################")
df.info()

# Inspección de unknown por columna contact
print("\nValores unknown detectado por columna:") 
columnas_categoricas = df.select_dtypes(include="object").columns
for col in columnas_categoricas:
    n_unknown = (df[col] == "unknown").sum()
    if n_unknown > 0:
        pct = n_unknown / len(df) * 100
        print(f"{col}: {n_unknown} valores 'unknown' ({pct:.2f}%)")

df["age"].describe()

print("\nDistribución de la variable sensible: Estado Civil\n", df["marital"].value_counts(normalize=True) * 100)
print("\nDistribución de la variable objetivo: Suscripción\n", df["y"].value_counts(normalize=True) * 100)
print("\nEstadísticas de edad\n", df["age"].describe())




################## PASO 2. LIMPIEZA CONSCIENTE DEL SESGO
cols_faltantes_genuinos = ["job", "education"]

# Ver el impacto de eliminar filas con nulos, ANTES de hacerlo
antes = df["marital"].value_counts(normalize=True) * 100

df_na = df.copy()
df_na[cols_faltantes_genuinos] = df_na[cols_faltantes_genuinos].replace("unknown", np.nan)
df_limpio = df_na.dropna(subset=cols_faltantes_genuinos).copy()

despues = df_limpio["marital"].value_counts(normalize=True) * 100

comparacion = pd.DataFrame({"Antes de limpiar (%)": antes, "Después de limpiar (%)": despues, "Diferencia": despues - antes}) #Construye 
#una tabla (Dataframe de resumen) combinando diccionarios de datos.
print("\nImpacto de de la limpieza en la variable marital (Estado Civil):\n",comparacion.round(2))

filas_antes, filas_despues = len(df), len(df_limpio)
pct_perdido = (1 - filas_despues / filas_antes) * 100
print(f"\nTotal de filas ANTES: {filas_antes}")
print(f"Total de filas DESPUÉS: {filas_despues}")
print(f"Porcentaje de filas descartadas: {pct_perdido:.2f}%")

bins_edad = [0, 25, 35, 45, 55, 65, 100]
etiquetas_edad = ["18-25", "26-35", "36-45", "46-55", "56-65", "66+"]
df_limpio["grupo_edad"] = pd.cut(df_limpio["age"], bins=bins_edad, labels=etiquetas_edad)

print("\n--- Distribución por grupo de edad (después de limpiar) ---")
print(df_limpio["grupo_edad"].value_counts(normalize=True).sort_index() * 100)

print("\n--- Tamaño real de cada subgrupo de edad (n) ---")
print(df_limpio["grupo_edad"].value_counts().sort_index())


################## PASO 3. TASA DE RESULTADO POSITIVO POR GRUPO (PARIDAD ESTADÍSTICA)
df["suscrito"] = (df["y"] == "yes").astype(int)
df_limpio["suscrito"] = (df_limpio["y"] == "yes").astype(int)

#Función tasa positiva
def tasa_positiva(data, columna_grupo): #Declara una función reusable con cuatro parámetros.
    resumen = data.groupby(columna_grupo, observed=True)["suscrito"].agg(["mean", "count"])
    resumen["mean"] = resumen["mean"] * 100
    resumen.columns = ["tasa_%", "n"]
    return resumen

tasa_estado_civil = tasa_positiva(df_limpio, "marital")
tasa_grupo_edad = tasa_positiva(df_limpio, "grupo_edad")

print("\nTasa de suscripción por estado civil:", tasa_estado_civil)
print("\nTasa de suscripción por grupo de edad:", tasa_grupo_edad)



#Métrica de disparidad (diferencia de paridad estadística):
disparidad_estado_civil = tasa_estado_civil["tasa_%"].max() - tasa_estado_civil["tasa_%"].min()
disparidad_edad = tasa_grupo_edad["tasa_%"].max() - tasa_grupo_edad["tasa_%"].min()

print(f"\nDisparidad por ecstado civil: {disparidad_estado_civil:.2f} puntos porcentuales")
print(f"Disparidad por grupo de edad: {disparidad_edad:.2f} puntos porcentuales")

tasa_global_completo = df["suscrito"].mean() * 100
tasa_global_limpio = df_limpio["suscrito"].mean() * 100
print(f"\nTasa de suscripción global — dataset completo (n={len(df)}): {tasa_global_completo:.2f}%")
print(f"Tasa de suscripción global — dataset limpio (n={len(df_limpio)}): {tasa_global_limpio:.2f}%")


################## PASO 4. VISUALIZACIÓN COMPARATIVA
# Matplotlib y Seaborn para visualizar gráficamente las diferencias en la tasa de ingresos altos (>50K) según sexo y raza, 
#facilitando la identificación visual de sesgos.
fig, ax = plt.subplots(1, 2, figsize=(14, 5)) 
# Crea una figura base (fig) que contendrá una cuadrícula de gráficos dividida en 1 fila y 2 columnas. 
#Retorna los ejes en un arreglo (ax), donde ax[0] es el gráfico de la izquierda y ax[1] el de la derecha.
# figsize: Establece las dimensiones de toda la figura en pulgadas: 12 de ancho por 5 de alto.

#Tasa por estado civil (Panel izquierdo)
tasa_estado_civil["tasa_%"].plot(kind="bar", ax=ax[0], color=["#4C72B0", "#DD8452", "#55A868"], edgecolor="black")
ax[0].set_title("Tasa de suscripción por Estado Civil", fontsize=12)
ax[0].set_ylabel("Tasa de éxito (%)")
ax[0].tick_params(axis="x", rotation=0)
for i, (tasa, n) in enumerate(zip(tasa_estado_civil["tasa_%"], tasa_estado_civil["n"])):
    ax[0].text(i, tasa + 0.5, f"n={n}", ha="center", fontsize=9)

#Tasa por grupo edad (Panel derecho)
tasa_grupo_edad["tasa_%"].plot(kind="bar", ax=ax[1], color="#C44E52", edgecolor="black")
ax[1].set_title("Tasa de suscripción por Grupo de Edad", fontsize=12)
ax[1].set_ylabel("Tasa de éxito (%)")
ax[1].tick_params(axis="x", rotation=45)
for i, (tasa, n) in enumerate(zip(tasa_grupo_edad["tasa_%"], tasa_grupo_edad["n"])):
    ax[1].text(i, tasa + 0.5, f"n={n}", ha="center", fontsize=9)

plt.tight_layout()
plt.savefig("chart1.png", dpi=110)
plt.show()


################## PASO 5. BÚSQUEDA DE PROXIES
crosstab_job_marital = pd.crosstab(df_limpio["job"], df_limpio["marital"], normalize="index") * 100

plt.figure(figsize=(8, 8))
sns.heatmap(crosstab_job_marital, annot=True, fmt=".1f", cmap="coolwarm", cbar_kws={"label": "Porcentaje (%)"})
plt.title("Relación entre Ocupación y Estado Civil (Detección de variable Proxy)", fontsize=13)
plt.xlabel("Estado civil")
plt.ylabel("Ocupación")
plt.tight_layout()
plt.savefig("chart2.png", dpi=110)
plt.show()

crosstab_job_edad = pd.crosstab(df_limpio["job"], df_limpio["grupo_edad"], normalize="index") * 100

plt.figure(figsize=(9, 8))
sns.heatmap(crosstab_job_edad, annot=True, fmt=".1f", cmap="coolwarm", cbar_kws={"label": "Porcentaje (%)"})
plt.title("Relación entre Ocupación y Grupo de Edad (Detección de variable Proxy)", fontsize=13)
plt.xlabel("Grupo de edad")
plt.ylabel("Ocupación")
plt.tight_layout()
plt.savefig("chart3.png", dpi=110)
plt.show()