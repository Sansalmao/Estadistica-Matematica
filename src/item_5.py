import pandas as pd
import statsmodels.formula.api as sm
import matplotlib.pyplot as plt

# Cargar los datos
datos = pd.read_csv('datos_.csv')

# Convertir variables categóricas a numéricas
datos = pd.get_dummies(datos, columns=['sex', 'smoker', 'region'], drop_first=True)

# Crear el modelo lineal
modelo = sm.ols('expenses ~ age + sex_male + bmi + children + smoker_yes + region_northwest + region_southeast + region_southwest', data=datos).fit()

# Función para predecir los gastos médicos
def predecir_gastos(age, sex_male, bmi, children, smoker_yes, region_northwest, region_southeast, region_southwest):

  # Crear un diccionario con los valores de las variables predictoras
  variables = {'age': age, 'sex_male': sex_male, 'bmi': bmi, 'children': children,
               'smoker_yes': smoker_yes, 'region_northwest': region_northwest,
               'region_southeast': region_southeast, 'region_southwest': region_southwest}
  # Crear un DataFrame con los valores de las variables predictoras
  df_prediccion = pd.DataFrame([variables])
  # Realizar la predicción utilizando el modelo
  prediccion = modelo.predict(df_prediccion)
  
  return prediccion[0]

# Ejemplo de uso de la función
gastos_predichos = predecir_gastos(age=30, sex_male=1, bmi=25, children=2, smoker_yes=0, 
                                   region_northwest=0, region_southeast=1, region_southwest=0)

print(f"Gastos médicos calculados: {gastos_predichos}\n")

predicciones = modelo.predict(datos)  # Obtener predicciones para todo el conjunto de datos

# Crear histograma de las predicciones
plt.hist(predicciones, bins=20) 
plt.xlabel("Gastos Médicos")
plt.ylabel("Frecuencia")
plt.title("Histograma de Gastos Médicos calculados")
plt.show()
