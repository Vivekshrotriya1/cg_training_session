import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

data = {
"Engine_Size": [1.2,1.5,1.8,2.0,2.2,1.3,1.6,2.4,2.0,1.4,1.7,2.5,1.8,2.2,1.5],
"Mileage": [90,70,60,50,40,85,65,30,45,80,55,25,50,35,75],
"Age": [8,6,5,4,3,7,6,2,4,7,5,1,3,2,6],
"Horsepower": [80,95,110,130,150,85,100,180,140,90,115,200,125,160,105],
"Price": [3.5,5,6,8,10,4,5.5,14,9,4.5,6.5,16,8.5,12,5.2]
}

df = pd.DataFrame(data)
print(df.head())
print(df.describe())
print(df.corr())
X = df[["Engine_Size","Mileage","Age","Horsepower"]]
y = df["Price"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model = LinearRegression()
model.fit(X_train, y_train)

print("Intercept:", model.intercept_)
print("Coefficients:", model.coef_)
y_pred = model.predict(X_test)

print("MAE:", mean_absolute_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))
new_car = pd.DataFrame([[2.0, 40, 3, 140]],
                       columns=["Engine_Size","Mileage","Age","Horsepower"])

predicted_price = model.predict(new_car)

print(f"Predicted Car Price: {predicted_price[0]:.2f} Lakhs")