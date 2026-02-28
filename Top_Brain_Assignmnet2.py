import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


df = pd.read_csv('multil_salary_pred_Sheet1.csv')


X = df[['Experience_years', 'Education_Level', 'Skills_Count', 'Performance_Rating']]
y = df['Salary_lpa']


model = LinearRegression()
model.fit(X, y)


b0 = model.intercept_
b1, b2, b3, b4 = model.coef_

print("Learned Multiple Regression Equation:")
print(f"Salary = {b0:.3f} + ({b1:.3f} × Experience) "
      f"+ ({b2:.3f} × Education) "
      f"+ ({b3:.3f} × Skills) "
      f"+ ({b4:.3f} × Performance)")



new_employee = [[6, 2, 8, 4]]
predicted_salary = model.predict(new_employee)

print("\nPredicted Salary:")
print(f"{predicted_salary[0]:.2f} LPA")


y_pred = model.predict(X)

plt.figure()
plt.scatter(y, y_pred)

plt.xlabel("Actual Salary (LPA)")
plt.ylabel("Predicted Salary (LPA)")
plt.title("Actual vs Predicted Salary (Multiple Linear Regression)")
plt.show()