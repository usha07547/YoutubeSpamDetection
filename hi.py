# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Load and clean dataset
df = sns.load_dataset('mpg')
df = df.dropna()

# Feature and target
X = df[['horsepower']]
y = df['mpg']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Polynomial features (degree 2)
poly = PolynomialFeatures(degree=2)
X_poly_train = poly.fit_transform(X_train)
X_poly_test = poly.transform(X_test)

# Train model
model = LinearRegression()
model.fit(X_poly_train, y_train)

# Predict on test data
y_pred = model.predict(X_poly_test)

# Visualization
X_fit = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
X_fit_poly = poly.transform(X_fit)
y_fit = model.predict(X_fit_poly)

plt.scatter(X, y, color='blue', label='Data')
plt.plot(X_fit, y_fit, color='red', linewidth=2, label='Polynomial Fit (Degree 2)')
plt.title('Polynomial Regression on Auto MPG Dataset')
plt.xlabel('Horsepower')
plt.ylabel('Miles Per Gallon (MPG)')
plt.legend()
plt.grid(True)
plt.show()

# Evaluation
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R^2 Score:", r2_score(y_test, y_pred))
