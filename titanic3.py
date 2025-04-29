# Import Libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Step 1: Load Dataset
df = pd.read_csv('titanic.csv')

# Step 2: Preprocess
df = df[['Age', 'Fare']].dropna()  # Use only Age and Fare columns, drop missing values

# Step 3: Select Feature and Target
X = df[['Age']]    # Notice: Only 1 feature -> Simple Linear Regression
y = df['Fare']

# Step 4: Split Dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Train Model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 6: Predict
y_pred = model.predict(X_test)

# Step 7: Evaluate
print(f"MAE: {mean_absolute_error(y_test, y_pred):.2f}")
print(f"MSE: {mean_squared_error(y_test, y_pred):.2f}")
print(f"R2 Score: {r2_score(y_test, y_pred):.2f}")

# Step 8: Plotting
plt.scatter(X_test, y_test, color='blue', label='Actual Fare')
plt.plot(X_test, y_pred, color='red', label='Predicted Fare')
plt.xlabel('Age')
plt.ylabel('Fare')
plt.title('Simple Linear Regression: Predicting Fare based on Age')
plt.legend()
plt.show()

# Step 9: Coefficients
print("Intercept:", model.intercept_)
print("Coefficient:", model.coef_[0])

