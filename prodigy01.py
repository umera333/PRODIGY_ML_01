import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Sample data (replace with your actual dataset)
data = {
    'bedrooms': [2, 3, 4, 3, 5],
    'bathrooms': [1, 2, 3, 2, 4],
    'square_footage': [1000, 1500, 2000, 1800, 2500],
    'price': [150000, 250000, 350000, 300000, 450000]
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Features and target
X = df[['bedrooms', 'bathrooms', 'square_footage']]
y = df['price']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)

# Evaluate
mse = mean_squared_error(y_test, predictions)
print("Mean Squared Error:", mse)

# Example prediction
example = pd.DataFrame({'bedrooms': [3], 'bathrooms': [2], 'square_footage': [1600]})
predicted_price = model.predict(example)
print("Predicted House Price:", predicted_price[0])

# Plotting Actual vs Predicted Prices
plt.figure(figsize=(10,6))
plt.scatter(y_test, predictions, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--')
plt.title('Actual vs Predicted House Prices')
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.show()

# Residual Plot (Actual - Predicted)
residuals = y_test - predictions
plt.figure(figsize=(10,6))
plt.scatter(predictions, residuals, color='green')
plt.axhline(y=0, color='red', linestyle='--')
plt.title('Residuals vs Predicted Prices')
plt.xlabel('Predicted Prices')
plt.ylabel('Residuals')
plt.show()

# Feature Importance Plot
feature_importance = model.coef_
plt.figure(figsize=(10,6))
plt.bar(X.columns, feature_importance, color='orange')
plt.title('Feature Importance')
plt.xlabel('Features')
plt.ylabel('Coefficient Value')
plt.show()