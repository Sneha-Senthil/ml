import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler

# Load dataset
df = pd.read_csv(r"C:\Users\sneha\Downloads\ML\ML\datasets\cardio.csv", delimiter=';')

# Convert age from days to years
df["age"] = df["age"] // 365

# --- EDA ---
print("\n--- Data Info ---")
print(df.info())

print("\n--- Summary Statistics ---")
print(df.describe())

print("\n--- Missing Values ---")
print(df.isna().sum())

# Fill missing values if needed
df.fillna(df.mean(numeric_only=True), inplace=True)

print("\n--- Missing Values After Filling ---")
print(df.isna().sum())

# Outlier removal using Z-score (only for numeric columns)
numeric_cols = df.select_dtypes(include=[np.number]).columns.drop(["id", "cardio"])
z_scores = np.abs((df[numeric_cols] - df[numeric_cols].mean()) / df[numeric_cols].std())
df_clean = df[(z_scores < 3).all(axis=1)]

print(f"\nOriginal shape: {df.shape}")
print(f"Shape after removing outliers: {df_clean.shape}")

# Define features and target
X = df_clean.drop(columns=["id", "cardio"])
y = df_clean["cardio"]

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled_df, y, test_size=0.2, random_state=42)

# Random Forest Model
rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
rf.fit(X_train, y_train)

# Predictions & Accuracy
y_pred = rf.predict(X_test)

print(f"\n--- Model Performance ---")
print(f"Random Forest Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["No Disease", "Has Disease"], yticklabels=["No Disease", "Has Disease"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Random Forest")
plt.show()

# Classification Report
print("\n--- Classification Report ---")
print(classification_report(y_test, y_pred, target_names=["No Disease", "Has Disease"]))

# Feature Importance Plot
plt.figure(figsize=(10, 6))
importances = pd.Series(rf.feature_importances_, index=X.columns)
importances.sort_values(ascending=True).plot(kind="barh", color="skyblue")
plt.title("Random Forest Feature Importances")
plt.xlabel("Importance")
plt.ylabel("Features")
plt.show()

# Actual vs Predicted Plot
plt.scatter(y_test, y_pred, alpha=0.5, c='purple')
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Actual vs Predicted - Random Forest")
plt.show()

# Residual Plot
residuals = y_test - y_pred
plt.scatter(y_pred, residuals, alpha=0.5)
plt.axhline(y=0, color='red', linestyle='--')
plt.xlabel("Predicted")
plt.ylabel("Residuals")
plt.title("Residuals vs Predicted - Random Forest")
plt.show()
