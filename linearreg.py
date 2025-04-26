
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

# Load dataset
cardio_df = pd.read_csv(r"C:\Users\sneha\Downloads\ML\ML\datasets\cardio.csv", delimiter=';')

# Optional: Drop 'id' column if present
if 'id' in cardio_df.columns:
    cardio_df.drop('id', axis=1, inplace=True)

# EDA
print("\n--- Data Info ---")
print(cardio_df.info())

print("\n--- Summary Statistics ---")
print(cardio_df.describe())

print("\n--- Missing Values ---")
print(cardio_df.isna().sum())

# Fill missing values in a sample numeric column (e.g., 'age') if needed
if 'age' in cardio_df.columns and cardio_df['age'].isna().sum() > 0:
    cardio_df['age'].fillna(cardio_df['age'].mean(), inplace=True)

print("\n--- Missing Values After Filling ---")
print(cardio_df.isna().sum())

# Outlier removal using Z-score
numeric_cols = cardio_df.select_dtypes(include=[np.number]).columns
z_scores = np.abs((cardio_df[numeric_cols] - cardio_df[numeric_cols].mean()) / cardio_df[numeric_cols].std())

cardio_clean = cardio_df[(z_scores < 3).all(axis=1)]

print(f"\nOriginal shape: {cardio_df.shape}")
print(f"Shape after removing outliers: {cardio_clean.shape}")

# Feature scaling
features = cardio_clean.drop('cardio', axis=1)
target = cardio_clean['cardio']

scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)
scaled_df = pd.DataFrame(scaled_features, columns=features.columns)

# Simple Linear Regression (not ideal for classification, but kept as requested)
X_train, X_test, y_train, y_test = train_test_split(scaled_df, target, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# Evaluation metrics
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n--- Model Performance ---")
print(f'MSE: {mse:.2f}')
print(f'RMSE: {rmse:.2f}')
print(f'MAE: {mae:.2f}')
print(f'R-squared: {r2:.2f}')

# Plot: Actual vs Predicted
plt.scatter(y_test, y_pred)
plt.xlabel('Actual Cardio')
plt.ylabel('Predicted Cardio')
plt.title('Actual vs Predicted Cardio')
plt.show()

# Residual plot
residuals = y_test - y_pred
plt.scatter(y_pred, residuals)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted Cardio')
plt.ylabel('Residuals')
plt.title('Residuals vs Predicted Values')
plt.show()

# Regression line plots for each numeric feature

sns.regplot(x=cardio_clean['age'], y=cardio_clean['cardio'], line_kws={"color": "red"})
plt.title(f'Regression Line: age vs Cardio')
plt.xlabel('age')
plt.ylabel('Cardio')

plt.tight_layout()
plt.show()


from statsmodels.stats.weightstats import ztest as ztest

# Z-test: Compare 'age' between people with cardio disease vs without
z_stat, p_value = ztest(
    cardio_clean[cardio_clean['cardio'] == 0]['age'],
    cardio_clean[cardio_clean['cardio'] == 1]['age']
)

print("\n--- Z-test: age vs cardio groups ---")
print(f"Z-statistic: {z_stat:.4f}")
print(f"P-value: {p_value:.4f}")
if p_value < 0.05:
    print("Result: Significant difference in 'age' between cardio and non-cardio groups.")
else:
    print("Result: No significant difference in 'age' between cardio and non-cardio groups.")
