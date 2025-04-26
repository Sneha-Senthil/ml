# Step 1: Import Libraries
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Step 2: Load Dataset
df = pd.read_csv(r"C:\Users\sneha\Downloads\ML\ML\datasets\abalone.csv")

# Step 3: Create Age Group (Target Variable)
df['agegroup'] = df['Rings'].apply(lambda x: "Young" if x <= 10 else "Old")

# Step 4: Encode Target Labels
encoder = LabelEncoder()
df['agegroup'] = encoder.fit_transform(df['agegroup'])  # Old=0, Young=1 (typically)

# Step 5: Feature Selection
X = df[['Height', 'Length', 'Diameter', 'Whole weight']]
y = df['agegroup']

# Step 6: Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 7: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Step 8: Model Training
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)

# Step 9: Predictions
y_pred = model.predict(X_test)

# Step 10: Evaluation Metrics
accuracy = accuracy_score(y_test, y_pred)
print(f"\n--- Model Performance ---")
print(f"Accuracy: {accuracy * 100:.2f}%")

# Step 11: Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['Old', 'Young'], yticklabels=['Old', 'Young'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - KNN")
plt.show()

# Step 12: Classification Report
print("\n--- Classification Report ---")
print(classification_report(y_test, y_pred, target_names=['Old', 'Young']))

# Step 13: Actual vs Predicted Plot
plt.scatter(y_test, y_pred, alpha=0.6, color='purple')
plt.xlabel("Actual Age Group")
plt.ylabel("Predicted Age Group")
plt.title("Actual vs Predicted - KNN")
plt.grid(True)
plt.show()

# Step 14: Residual Plot
residuals = y_test - y_pred
plt.scatter(y_pred, residuals, alpha=0.6, color='orange')
plt.axhline(y=0, color='red', linestyle='--')
plt.xlabel("Predicted Age Group")
plt.ylabel("Residuals")
plt.title("Residuals vs Predicted - KNN")
plt.show()
