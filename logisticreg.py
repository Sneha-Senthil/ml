import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load dataset
df = pd.read_csv(r"C:\Users\sneha\Downloads\ML\ML\datasets\abalone.csv")

# EDA
print("\n--- Data Info ---")
print(df.info())

print("\n--- Summary Statistics ---")
print(df.describe())

print("\n--- Missing Values ---")
print(df.isna().sum())

# Fill missing values if needed (if there are any in numerical columns)
#df[].fillna(df[].mean(), inplace=True)

print("\n--- Missing Values After Filling ---")
print(df.isna().sum())

# Outlier removal using Z-score (for numerical columns)
numeric_cols = df.select_dtypes(include=[np.number]).columns
z_scores = np.abs((df[numeric_cols] - df[numeric_cols].mean()) / df[numeric_cols].std())

# Remove rows where any z-score is greater than 3 (commonly used threshold)
df_clean = df[(z_scores < 3).all(axis=1)]

print(f"\nOriginal shape: {df.shape}")
print(f"Shape after removing outliers: {df_clean.shape}")

# Encoding target 'Sex' (categorical to numerical)
encoder = LabelEncoder()
df_clean['Sex'] = encoder.fit_transform(df_clean['Sex'])

# Define features (X) and target (y)
X = df_clean.drop('Sex', axis=1)  # All columns except 'Sex'
y = df_clean['Sex']  # Target column 'Sex'

# Feature scaling
scaler = StandardScaler()
scaled_features = scaler.fit_transform(X)  # Scaling the features (X)
scaled_df = pd.DataFrame(scaled_features, columns=X.columns)  # Creating a scaled DataFrame

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(scaled_df, y, test_size=0.2, random_state=42)

# Create and train the Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
print(f"\n--- Model Performance ---")
print(f"Accuracy: {accuracy * 100:.2f}%")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Classification Report
print("\n--- Classification Report ---")
print(classification_report(y_test, y_pred))

# Plot: Actual vs Predicted
plt.scatter(y_test, y_pred)
plt.xlabel('Actual Sex')
plt.ylabel('Predicted Sex')
plt.title('Actual vs Predicted Sex')
plt.show()

# Residual plot (for logistic regression, this isn't typical but can be used for insights)
residuals = y_test - y_pred
plt.scatter(y_pred, residuals)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted Sex')
plt.ylabel('Residuals')
plt.title('Residuals vs Predicted Sex')
plt.show()
