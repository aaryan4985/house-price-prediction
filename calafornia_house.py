import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.datasets import fetch_california_housing

# Load the California housing dataset
california = fetch_california_housing()
data = pd.DataFrame(california.data, columns=california.feature_names)
data['Price'] = california.target

# Step 1: Data Exploration
print("\nData Information:")
print(data.info())

print("\nDescriptive Statistics:")
print(data.describe())

print("\nCorrelation Matrix:")
print(data.corr()['Price'].sort_values(ascending=False))

# Step 2: Splitting the Dataset
X = data.drop("Price", axis=1)
y = data["Price"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\nShape of Training Data:", X_train.shape)
print("Shape of Testing Data:", X_test.shape)

# Step 3: Train the Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)

print("\nModel Coefficients:")
print(model.coef_)
print("\nIntercept:", model.intercept_)

# Step 4: Make Predictions
y_pred = model.predict(X_test)
print("\nFirst 5 Predictions:")
print(y_pred[:5])

print("\nFirst 5 Actual Prices:")
print(y_test[:5].values)

# Step 5: Evaluate the Model
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print("\nModel Evaluation:")
print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)

# Step 6: Visualization
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color="blue", alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color="red", lw=2)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted Housing Prices")
plt.show()
