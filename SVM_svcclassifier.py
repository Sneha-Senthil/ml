import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load dataset with correct delimiter
df = pd.read_csv(r"C:\Users\sneha\Downloads\ML\ML\datasets\cardio.csv", delimiter=';')

# Optional: Drop 'id' column if present
if 'id' in df.columns:
    df.drop('id', axis=1, inplace=True)

# EDA
print("\n--- Data Info ---")
print(df.info())

print("\n--- Summary Statistics ---")
print(df.describe())

print("\n--- Missing Values ---")
print(df.isna().sum())

# Fill missing values if needed (fill numeric columns with mean)
if 'age' in df.columns and df['age'].isna().sum() > 0:
    df['age'].fillna(df['age'].mean(), inplace=True)

print("\n--- Missing Values After Filling ---")
print(df.isna().sum())

# Outlier removal using Z-score
numeric_cols = df.select_dtypes(include=[np.number]).columns
z_scores = np.abs((df[numeric_cols] - df[numeric_cols].mean()) / df[numeric_cols].std())

df_clean = df[(z_scores < 3).all(axis=1)]

print(f"\nOriginal shape: {df.shape}")
print(f"Shape after removing outliers: {df_clean.shape}")

# Feature selection (X = features, y = target)
X = df_clean[['age', 'height', 'weight', 'ap_hi', 'ap_lo']]
y = df_clean['cardio']

# Step 1: Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 2: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Step 3: Model Training (SVM with RBF kernel)
model = SVC(kernel="rbf", C=1.0, gamma='scale')
model.fit(X_train, y_train)

# Step 4: Predictions
y_pred = model.predict(X_test)

# Step 5: Evaluation Metrics
accuracy = accuracy_score(y_test, y_pred)
print(f"\nSVM Classifier Accuracy: {accuracy:.2f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['No Cardio', 'Cardio'], yticklabels=['No Cardio', 'Cardio'])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Step 6: Actual vs Predicted Plot
plt.scatter(y_test, y_pred, alpha=0.6, color='purple')
plt.xlabel("Actual Cardio")
plt.ylabel("Predicted Cardio")
plt.title("Actual vs Predicted - SVM")
plt.grid(True)
plt.show()

# Step 7: Residual Plot
residuals = y_test - y_pred
plt.scatter(y_pred, residuals, alpha=0.6, color='orange')
plt.axhline(y=0, color='red', linestyle='--')
plt.xlabel("Predicted Cardio")
plt.ylabel("Residuals")
plt.title("Residuals vs Predicted - SVM")
plt.show()
