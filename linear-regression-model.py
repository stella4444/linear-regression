# Install the 'scikit-learn' and 'numpy' libraries. They can be installed via pip with:
# 'pip install scikit-learn numpy'
# Import necessary libraries
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Example data
# X represents the input features (independent variable)
# y represents the target variable (dependent variable)
X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)  # 2D array for scikit-learn
y = np.array([2, 4, 5, 4, 5])

# Create a Linear Regression model
model = LinearRegression()

# Fit the model to the data
model.fit(X, y)

# Make predictions
y_pred = model.predict(X)

# Print out the results
print(f"Intercept: {model.intercept_}")
print(f"Coefficient: {model.coef_[0]}")

# Plotting the results
plt.scatter(X, y, color='blue', label='Data points')
plt.plot(X, y_pred, color='red', label='Fitted line')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Simple Linear Regression')
plt.legend()
plt.show()
