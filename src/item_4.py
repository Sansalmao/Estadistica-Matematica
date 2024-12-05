# Importar librerías
import pandas as pd
import statsmodels.formula.api as sm
from statsmodels.stats.anova import anova_lm
from tabulate import tabulate

# Cargar los datos
datos = pd.read_csv('datos_.csv')

# Realizar el ANOVA
modelo = sm.ols('bmi ~ region', data=datos).fit()
anova_tabla = anova_lm(modelo, typ=2)

anova_tabla = anova_tabla.fillna(0) 

# Imprimir la tabla con título
print("\nResultados:\n")
tabla_formateada = tabulate(anova_tabla, headers='keys', tablefmt='psql')
print(tabla_formateada)

# Interpretar los resultados
valor_p = anova_tabla.iloc[0]['PR(>F)']
nivel_significancia = 0.05

if valor_p < nivel_significancia:
    print("\nSe rechaza la hipótesis nula.")
    print("Existe evidencia estadísticamente significativa para afirmar que el promedio de bmi no es igual en las cuatro regiones.")
else:
    print("\nNo se rechaza la hipótesis nula.")
    print("No hay suficiente evidencia para afirmar que el promedio de bmi es diferente entre las regiones.")
