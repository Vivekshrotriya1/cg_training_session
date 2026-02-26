import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


df = pd.read_csv('salary_lpa.csv')


X = df[['Experience_years']]
y = df['Salary_lpa']


model = LinearRegression()
model.fit(X, y)


m = model.coef_[0]
b = model.intercept_

print("Learned Regression Equation:")
print(f"Salary = {m:.3f} × Experience + {b:.3f}")


new_experience = [[7]]  
predicted_salary = model.predict(new_experience)

print("\nPredicted Salary for 7 years experience:")
print(f"{predicted_salary[0]:.2f} LPA")


x_range = np.linspace(X.min(), X.max(), 100)
y_line = model.predict(x_range)

plt.figure()
plt.scatter(X, y)
plt.plot(x_range, y_line)

plt.xlabel("Experience (Years)")
plt.ylabel("Salary (LPA)")
plt.title("Salary vs Experience (Linear Regression)")
plt.show()