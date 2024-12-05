import pandas as pd
import statsmodels.formula.api as sm
from tabulate import tabulate

# Cargar los datos
datos = pd.read_csv('datos_.csv')

# Convertir variables categóricas a numéricas usando one-hot encoding
datos = pd.get_dummies(datos, columns=['sex', 'smoker', 'region'], drop_first=True)

# Crear el modelo lineal
modelo = sm.ols('expenses ~ age + sex_male + bmi + children + smoker_yes + region_northwest + region_southeast + region_southwest', data=datos).fit()

# Nivel de significancia
nivel_significancia = 0.05

# Obtener la tabla de resultados del modelo
resultados = modelo.summary2().tables[1] 

# Personalizar la columna "Rechazar H0"
resultados['Rechazar H0'] = resultados.apply(lambda fila: f"Se rechaza H0 porque p ({fila['P>|t|']:.3f}) < {nivel_significancia}" if fila['P>|t|'] < nivel_significancia else f"No se rechaza H0 porque p ({fila['P>|t|']:.3f}) >= {nivel_significancia}", axis=1)

# Formatear los resultados como una lista ordenada
headers = [" ", "Coef.", "Std.Err.", "t", "P>|t|", "[0.025", "0.975]", "Rechazar H0"]  
tabla_formateada = tabulate(resultados, headers=headers, tablefmt="plain")  

# Imprimir los resultados con el título "Resultados"
print("\nResultados:\n")  
print(tabla_formateada)
