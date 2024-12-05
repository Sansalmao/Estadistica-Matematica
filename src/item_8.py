import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder


data = pd.read_csv("./src/datos_.csv")
le = LabelEncoder()

data["sex_aux"] = le.fit_transform(data["sex"])
data["smoker_aux"] = le.fit_transform(data["smoker"])
data["region_aux"] = le.fit_transform(data["region"])

X = data[["age", "sex_aux", "bmi", "children", "smoker_aux", "region_aux"]]
y = data["expenses"]

model = LinearRegression()
model.fit(X, y)

y_pred = model.predict(X)
residuals = y - y_pred

plt.scatter(y_pred, residuals)
plt.xlabel("Valores ajustados")
plt.ylabel("Residuos")
plt.title("Gráfico de residuos vs. valores ajustados")
plt.show()
