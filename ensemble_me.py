import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

# Load dataset
data = pd.read_csv("bill_authentication.csv")

# --- EDA ---
print("\n--- Data Info ---")
print(data.info())

print("\n--- Summary Statistics ---")
print(data.describe())

print("\n--- Missing Values ---")
print(data.isna().sum())

# Check class balance
sns.countplot(data['Class'])
plt.title("Class Distribution (0: Fake, 1: Genuine)")
plt.show()

# Correlation matrix
plt.figure(figsize=(8, 6))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Feature Correlation Matrix")
plt.show()

# Boxplots to check for outliers
data.drop(columns=["Class"]).plot(kind='box', subplots=True, layout=(2,2), figsize=(10,6), sharex=False, sharey=False)
plt.tight_layout()
plt.show()

# --- Preprocessing ---
X = data.drop(columns=["Class"]).values
y = data["Class"].values

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# --- Unified Evaluation Function ---
def evaluate_model(name, model):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print(f"\n{name} Accuracy: {accuracy_score(y_test, y_pred)*100:.2f}%")
    print("Classification Report:\n", classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Fake", "Genuine"], yticklabels=["Fake", "Genuine"])
    plt.title(f"{name} - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

# --- Bagging ---
evaluate_model("Random Forest (Bagging)", RandomForestClassifier(n_estimators=100, random_state=42))

# --- Boosting ---
evaluate_model("XGBoost (Boosting)", XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42))

# --- Stacking ---
base_models = [
    ('knn', KNeighborsClassifier(n_neighbors=3)),
    ('dt', DecisionTreeClassifier(max_depth=3))
]
stack_model = StackingClassifier(estimators=base_models, final_estimator=SVC(kernel='linear'))
evaluate_model("Stacking (KNN + DT → SVM)", stack_model)
