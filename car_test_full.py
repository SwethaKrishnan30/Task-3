import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb

# Load the data
data = pd.read_csv('C:/Users/Joshua/Downloads/car data.csv')

# Data Preprocessing
data = data.dropna()
label_encoder = LabelEncoder()
data['Fuel_Type'] = label_encoder.fit_transform(data['Fuel_Type'])
data['Selling_type'] = label_encoder.fit_transform(data['Selling_type'])
data['Transmission'] = label_encoder.fit_transform(data['Transmission'])

X = data.drop(['Car_Name', 'Selling_Price'], axis=1)
y = data['Selling_Price']

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Feature Engineering
X['Car_Age'] = 2024 - X['Year']
X = X.drop(['Year'], axis=1)

# Model Building
model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42)
model.fit(X_scaled, y)

# Predicting for the whole dataset
y_pred = model.predict(X_scaled)

# Adding predictions to the original dataset
data['Predicted_Selling_Price'] = y_pred

# Model Evaluation Metrics
mae = mean_absolute_error(y, y_pred)
mse = mean_squared_error(y, y_pred)
rmse = mean_squared_error(y, y_pred, squared=False)
r2 = r2_score(y, y_pred)

print(f'Mean Absolute Error: {mae:.2f}')
print(f'Mean Squared Error: {mse:.2f}')
print(f'Root Mean Squared Error: {rmse:.2f}')
print(f'R-squared: {r2:.2f}')

# Visualizations
plt.figure(figsize=(14, 6))

# Scatter plot of actual vs predicted prices
plt.subplot(1, 2, 1)
plt.scatter(y, y_pred, alpha=0.6, color='b')
plt.plot(y, y, color='r', linestyle='--')
plt.title('Actual vs Predicted Selling Prices')
plt.xlabel('Actual Selling Price')
plt.ylabel('Predicted Selling Price')
plt.grid(True)

# Distribution plot of residuals
plt.subplot(1, 2, 2)
residuals = y - y_pred
sns.histplot(residuals, kde=True, color='g', bins=25)
plt.title('Distribution of Residuals')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.grid(True)

plt.tight_layout()
plt.show()

# Feature Importance Plot
plt.figure(figsize=(10, 8))
feature_importance = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
sns.barplot(x=feature_importance, y=feature_importance.index, palette='viridis')
plt.title('Feature Importance')
plt.xlabel('Importance Score')
plt.ylabel('Features')
plt.grid(True)
plt.show()

# Displaying predictions
print(data[['Car_Name', 'Selling_Price', 'Predicted_Selling_Price']].head(10))  # Displaying first 10 rows for illustration

# Optionally, you can save the predictions to a new CSV file
data.to_csv('predicted_car_prices.csv', index=False)
