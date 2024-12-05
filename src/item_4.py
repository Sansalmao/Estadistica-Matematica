import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import f_regression

data = pd.read_csv("./src/datos_.csv")


X = data[["region"]]
y = data["bmi"]
vec = DictVectorizer(sparse=False)
X = vec.fit_transform(X.to_dict("records"))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LinearRegression()
model.fit(X_train, y_train)

F_statistic, p_values = f_regression(X, y)

print("F-statistic:", F_statistic)
print("p-values:", p_values)
