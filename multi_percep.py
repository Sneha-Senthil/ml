import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load the dataset
digits = load_digits()
X, y = digits.data, digits.target  # Flattened images (64 features per image)

# Convert to DataFrame for EDA
df = pd.DataFrame(X)
df['target'] = y

# --------------------------- EDA ---------------------------
print("\n--- Dataset Shape ---")
print(df.shape)

print("\n--- First 5 Rows ---")
print(df.head())

print("\n--- Data Info ---")
print(df.info())

print("\n--- Summary Statistics ---")
print(df.describe())

print("\n--- Missing Values ---")
print(df.isna().sum().sum())  # Should be 0

# Visualize the count of digits
sns.countplot(x='target', data=df)
plt.title("Distribution of Digits")
plt.xlabel("Digit")
plt.ylabel("Count")
plt.show()

# Visualize one sample image
plt.imshow(digits.images[0], cmap='gray')
plt.title(f"Example Digit: {digits.target[0]}")
plt.axis("off")
plt.show()

# ------------------ Feature Selection ---------------------
features = df.drop('target', axis=1)
target = df['target']

# ------------------ Feature Scaling -----------------------
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# ------------------ Train-Test Split ----------------------
X_train, X_test, y_train, y_test = train_test_split(features_scaled, target, test_size=0.2, random_state=42)

# ------------------ Model Training ------------------------
mlp = MLPClassifier(hidden_layer_sizes=(128, 64), activation='relu', solver='adam', max_iter=500, random_state=42)
mlp.fit(X_train, y_train)

# ------------------ Predictions ---------------------------
y_pred = mlp.predict(X_test)

# ------------------ Evaluation ----------------------------
accuracy = accuracy_score(y_test, y_pred)
print(f"\n--- Model Performance ---")
print(f"Test Accuracy: {accuracy:.4f}")

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=digits.target_names, yticklabels=digits.target_names)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Visualize prediction for a test sample
index = 2
plt.imshow(X_test[index].reshape(8, 8), cmap='gray')
plt.title(f"Predicted Label: {y_pred[index]}")
plt.axis("off")
plt.show()
