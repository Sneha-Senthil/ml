import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load dataset
df = pd.read_csv(r"C:\Users\sneha\Downloads\ML\ML\datasets\abalone.csv")

# EDA
print("\n--- Data Info ---")
print(df.info())

print("\n--- Summary Statistics ---")
print(df.describe())

print("\n--- Missing Values ---")
print(df.isna().sum())

# Fill missing values if needed (if any exist)
# df.fillna(df.mean(numeric_only=True), inplace=True)

print("\n--- Missing Values After Filling ---")
print(df.isna().sum())

# Outlier removal using Z-score (for numerical columns only)
numeric_cols = df.select_dtypes(include=[np.number]).columns
z_scores = np.abs((df[numeric_cols] - df[numeric_cols].mean()) / df[numeric_cols].std())
df_clean = df[(z_scores < 3).all(axis=1)]

print(f"\nOriginal shape: {df.shape}")
print(f"Shape after removing outliers: {df_clean.shape}")

# Encode the target column 'Sex'
encoder = LabelEncoder()
df_clean['Sex'] = encoder.fit_transform(df_clean['Sex'])  # M, F, I → 0, 1, 2

# Features & Target
X = df_clean[['Height', 'Length']]  # Features used for Naive Bayes
y = df_clean['Sex']  # Encoded target

# Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled_df, y, test_size=0.2, random_state=42)

# Model Training
model = GaussianNB()
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\n--- Model Performance ---")
print(f"Naive Bayes Accuracy: {accuracy * 100:.2f}%")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", xticklabels=encoder.classes_, yticklabels=encoder.classes_)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Naive Bayes")
plt.show()

# Classification Report
print("\n--- Classification Report ---")
print(classification_report(y_test, y_pred, target_names=encoder.classes_))

# Scatter Plot - Predictions
plt.figure(figsize=(8, 5))
plt.scatter(X_test['Length'], X_test['Height'], c=y_pred, cmap='viridis', alpha=0.7)
plt.xlabel("Length")
plt.ylabel("Height")
plt.title("Naïve Bayes Classification of Abalone Sex")
plt.colorbar(label="Predicted Class (0: F, 1: I, 2: M)")
plt.show()

# Residual Plot
residuals = y_test - y_pred
plt.scatter(y_pred, residuals, alpha=0.6)
plt.axhline(y=0, color='red', linestyle='--')
plt.xlabel("Predicted Sex")
plt.ylabel("Residuals")
plt.title("Residuals vs Predicted Sex - Naive Bayes")
plt.show()
