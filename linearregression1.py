import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# # Step 1: Create dataset
# data = {
#     "Area_sqft": [600, 800, 1200, 1400, 1600, 1800, 2000, 2200, 2500, 2800],
#     "Price_lakhs": [30, 40, 50, 60, 68, 75, 85, 95, 105, 120]
# }

# df = pd.DataFrame(data)

# print("Dataset:")
# print(df)

# # Step 2: Define features (X) and target (y)
# X = df[["Area_sqft"]]
# y = df["Price_lakhs"]

# # Step 3: Train-test split
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42
# )

# # Step 4: Train the model
# model = LinearRegression()
# model.fit(X_train, y_train)

# # Step 5: Model parameters
# print("\nSlope (m):", model.coef_[0])
# print("Intercept (b):", model.intercept_)

# # Step 6: Predictions on test data
# y_pred = model.predict(X_test)

# print("\nActual vs Predicted:")
# for actual, pred in zip(y_test.values, y_pred):
#     print(f"Actual: {actual:.2f}, Predicted: {pred:.2f}")

# # Step 7: Model evaluation
# mae = mean_absolute_error(y_test, y_pred)
# r2 = r2_score(y_test, y_pred)

# print("\nMean Absolute Error:", mae)
# print("R2 Score:", r2)

# # Step 8: Predict price for new house
# new_area = pd.DataFrame({"Area_sqft": [1800]})
# predicted_price = model.predict(new_area)

# print(f"\nPredicted price for 1800 sqft house: Rs {predicted_price[0]:.2f} lakhs")

# # Step 9: Visualization
# plt.scatter(X, y, label="Actual Data")
# plt.plot(X, model.predict(X), label="Regression Line")
# plt.xlabel("Area (sqft)")
# plt.ylabel("Price (Lakhs)")
# plt.title("House Price Prediction using Linear Regression")
# plt.legend()
# plt.show()


data = {
    "Area_sqft": [800,1000,1200,1500,1800,2000,2200,2500,900,1600,1400,2100],
    "Bedrooms": [2,2,3,3,4,4,4,5,2,3,3,4],
    "Age_years": [15,10,8,5,4,3,2,1,12,6,7,3],
    "Distance_km": [12,10,8,6,5,4,3,2,11,7,9,4],
    "Price_lakhs": [40,50,62,75,90,105,120,140,45,80,70,110]
}

df=pd.DataFrame(data)

print("Dataset: ")
print(df)

#step3

x=df[["Area_sqft","Bedrooms","Age_years","Distance_km"]]
y=df["Price_lakhs"]

X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

model=LinearRegression()
model.fit(X_train,y_train)

y_pred=model.predict(X_test)

print("\nActual vs Predicted:")
for actual, pred in zip(y_test.values, y_pred):
    print(f"Actual: {actual:.2f}, Predicted: {pred:.2f}")

print("\n Mean Absolute Error: ", mean_absolute_error(y_test, y_pred))
print("R2 Score: ",r2_score(y_test,y_pred))


new_house_data={
    "Area_sqft": [1700],
    "Bedrooms":[3],
    "Age_years":[5],
    "Distance_km":[6]
}

new_house_df=pd.DataFrame(new_house_data)

predicted_price =model.predict(new_house_df)
print("Predicted Price for New House: Rs. {:.2f} lakhs".format(predicted_price[0]))