# ?????????????????????????????????
import pandas as pd
import statsmodels.formula.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

# Cargar los datos
data = pd.read_csv("./src/datos_.csv")

# Convertir variables categóricas a numéricas usando one-hot encoding
data = pd.get_dummies(data, columns=["sex", "smoker", "region"], drop_first=True)

# Crear el modelo lineal
model = sm.ols(
    "expenses ~ age + sex_male + bmi + children + smoker_yes + region_northwest + region_southeast + region_southwest",
    data=data,
).fit()

# Mostrar los resultados del modelo
print(model.summary())

# Diagnóstico del modelo (opcional)
# Gráfico de residuos vs valores ajustados
plt.figure(figsize=(8, 6))
sns.residplot(x=model.fittedvalues, y=model.resid)
plt.title("Gráfico de Residuos vs Valores Ajustados")
plt.xlabel("Valores Ajustados")
plt.ylabel("Residuos")
plt.show()

# Histograma de los residuos
plt.figure(figsize=(8, 6))
sns.histplot(model.resid, kde=True)
plt.title("Histograma de los Residuos")
plt.xlabel("Residuos")
plt.show()
