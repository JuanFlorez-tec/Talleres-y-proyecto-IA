import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option("display.max_columns", None)
sns.set_style("whitegrid")


#1. CARGAR DATASET

df = pd.read_csv("bank-full.csv", sep=";")

print("\n========== INFORMACIÓN INICIAL ==========")

print("\nDimensiones:")
print(df.shape)

print("\nInformación:")
print(df.info())

print("\nValores nulos:")
print(df.isnull().sum())

#2. EXPLORACIÓN

print("\n========== EXPLORACIÓN ==========")

print("\nDistribución de marital:")
print(df["marital"].value_counts())

print("\nDistribución de job:")
print(df["job"].value_counts())

print("\nVariable objetivo:")
print(df["y"].value_counts())

print("\nValores unknown:")
for columna in df.columns:
    cantidad = (df[columna] == "unknown").sum()

    if cantidad > 0:
        print(f"{columna}: {cantidad}")


#3. CREAR GRUPOS DE EDAD

def grupo_edad(edad):
    if edad <= 30:
        return "<=30"
    elif edad <= 50:
        return "31-50"
    else:
        return ">50"


df["grupo_edad"] = df["age"].apply(grupo_edad)

print("\nDistribución por grupo de edad:")
print(df["grupo_edad"].value_counts())

print("\nPorcentaje por grupo de edad:")
print(df["grupo_edad"].value_counts(normalize=True) * 100)

# 4. LIMPIEZA

df_limpio_base = df.copy()

df_limpio_base = df_limpio_base.replace("unknown", np.nan)

print("\nValores nulos después de convertir unknown:")
print(df_limpio_base.isnull().sum())

#5. IMPACTO DE DROPNA

antes = (
    df_limpio_base["grupo_edad"]
    .value_counts(normalize=True)
    .sort_index()
)

df_limpio = df_limpio_base.dropna()

despues = (
    df_limpio["grupo_edad"]
    .value_counts(normalize=True)
    .sort_index()
)

comparacion = pd.DataFrame({
    "antes": antes,
    "despues": despues
})

comparacion["cambio"] = (
    comparacion["despues"] -
    comparacion["antes"]
)

print("\nComparación antes/después de dropna:")
print(comparacion)

#6. TASA POSITIVA

def tasa_positiva(
    data,
    columna_grupo,
    columna_objetivo,
    valor_positivo
):
    return (
        data.groupby(columna_grupo)[columna_objetivo]
        .apply(lambda x: (x == valor_positivo).mean())
    )


tasa_por_edad = tasa_positiva(
    df_limpio,
    "grupo_edad",
    "y",
    "yes"
)

print("\nTasa positiva por grupo de edad:")
print(tasa_por_edad)

#7. DISPARIDAD

disparidad_edad = (
    tasa_por_edad.max() -
    tasa_por_edad.min()
)

print(
    f"\nDisparidad por grupo de edad: "
    f"{disparidad_edad:.3f}"
)

#8. VISUALIZACIÓN

plt.figure(figsize=(8, 5))

tasa_por_edad.plot(kind="bar")

plt.title("Tasa de suscripción por grupo de edad")
plt.xlabel("Grupo de edad")
plt.ylabel("Proporción de suscripciones")

plt.xticks(rotation=0)
plt.tight_layout()

plt.show()

#9. BÚSQUEDA DE PROXY

crosstab_job_edad = pd.crosstab(
    df_limpio["job"],
    df_limpio["grupo_edad"],
    normalize="index"
)

print("\nCrosstab job vs grupo de edad:")
print(crosstab_job_edad)

#10. VISUALIZACIÓN DEL PROXY

crosstab_job_edad.plot(
    kind="bar",
    figsize=(12, 6)
)

plt.title(
    "Distribución de grupos de edad por ocupación"
)

plt.xlabel("Ocupación")
plt.ylabel("Proporción")

plt.xticks(rotation=45, ha="right")
plt.tight_layout()

plt.show()


#11. DUPLICADOS

duplicados = df.duplicated().sum()

print(
    f"\nCantidad de registros duplicados: "
    f"{duplicados}"
)

#12. OUTLIERS / EDAD

print("\nEstadísticas de edad:")
print(df["age"].describe())

plt.figure(figsize=(8, 4))

sns.boxplot(x=df["age"])

plt.title("Distribución de edad")
plt.xlabel("Edad")

plt.tight_layout()

plt.show()