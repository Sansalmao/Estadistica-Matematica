import pandas as pd
import statsmodels.formula.api as sm

data = pd.read_csv("./src/datos_.csv")

data = pd.get_dummies(data, columns=["sex", "smoker", "region"], drop_first=True)

model = sm.ols(
    "expenses ~ age + sex_male + bmi + children + smoker_yes + region_northwest + region_southeast + region_southwest",
    data=data,
).fit()

print(model.summary())

significance_level = 0.05

for coefficient in model.pvalues.index:
    p_valor = model.pvalues[coefficient]
    if p_valor < significance_level:
        print(f"\nSe rechaza H0 para el coeficiente de '{coefficient}'.")
        print(
            f"Hay evidencia de que '{coefficient}' tiene un efecto significativo en 'expenses'."
        )
    else:
        print(f"\nNo se rechaza H0 para el coeficiente de '{coefficient}'.")
        print(
            f"No hay suficiente evidencia para afirmar que '{coefficient}' tiene un efecto significativo en 'expenses'."
        )
