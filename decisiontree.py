import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler

# Load dataset
df = pd.read_csv(r"C:\Users\sneha\Downloads\ML\ML\datasets\cardio.csv", delimiter=';')

# Convert age from days to years
#df["age"] = df["age"] // 365

# --- EDA ---
print("\n--- Data Info ---")
print(df.info())

print("\n--- Summary Statistics ---")
print(df.describe())

print("\n--- Missing Values ---")
print(df.isna().sum())

# Fill missing values if needed
#df.fillna(df.mean(numeric_only=True), inplace=True)

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

# Decision Tree Model
dt = DecisionTreeClassifier(max_depth=3, random_state=42)
dt.fit(X_train, y_train)

# Predictions & Accuracy
y_pred = dt.predict(X_test)

print(f"\n--- Model Performance ---")
print(f"Decision Tree Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Oranges")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Decision Tree")
plt.show()

# Classification Report
print("\n--- Classification Report ---")
print(classification_report(y_test, y_pred))

# Decision Tree Plot
plt.figure(figsize=(14, 7))
plot_tree(dt, feature_names=X.columns, class_names=["No Disease", "Has Disease"], filled=True)
plt.title("Decision Tree Visualization")
plt.show()

# Actual vs Predicted Plot
plt.scatter(y_test, y_pred)
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Actual vs Predicted - Decision Tree")
plt.show()

# Residual Plot
residuals = y_test - y_pred
plt.scatter(y_pred, residuals)
plt.axhline(y=0, color='red', linestyle='--')
plt.xlabel("Predicted")
plt.ylabel("Residuals")
plt.title("Residuals vs Predicted - Decision Tree")
plt.show()



'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler

# Load dataset
df = pd.read_csv(r"C:\Users\sneha\Downloads\ML\ML\datasets\cardio.csv", delimiter=';')

# --- EDA ---
print("\n--- Data Info ---")
print(df.info())

print("\n--- Summary Statistics ---")
print(df.describe())

print("\n--- Missing Values ---")
print(df.isna().sum())

# --- Outlier Removal ---
numeric_cols = df.select_dtypes(include=[np.number]).columns.drop(["id", "cardio"])
z_scores = np.abs((df[numeric_cols] - df[numeric_cols].mean()) / df[numeric_cols].std())
df_clean = df[(z_scores < 3).all(axis=1)]

print(f"\nOriginal shape: {df.shape}")
print(f"Shape after removing outliers: {df_clean.shape}")

# --- Features & Target ---
X = df_clean.drop(columns=["id", "cardio"])
y = df_clean["cardio"]

# --- Feature Scaling ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

# --- Train-test Split ---
X_train, X_test, y_train, y_test = train_test_split(X_scaled_df, y, test_size=0.2, random_state=42)

# -----------------------------------------------------
# 🔹 PRE-PRUNING: Apply max_depth, min_samples_split, etc.
# -----------------------------------------------------
prepruned_dt = DecisionTreeClassifier(
    max_depth=5,
    min_samples_split=50,
    min_samples_leaf=25,
    max_leaf_nodes=10,
    random_state=42
)
prepruned_dt.fit(X_train, y_train)

# --- Predictions & Evaluation ---
y_pred_pre = prepruned_dt.predict(X_test)
print(f"\n--- Pre-Pruned Model Performance ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred_pre) * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_pre))

# --- Plot Pre-pruned Tree ---
plt.figure(figsize=(12, 6))
plot_tree(prepruned_dt, feature_names=X.columns, class_names=["No Disease", "Has Disease"], filled=True)
plt.title("Pre-Pruned Decision Tree")
plt.show()

# -----------------------------------------------------
# 🔹 POST-PRUNING: Cost-complexity pruning (ccp_alpha)
# -----------------------------------------------------
path = DecisionTreeClassifier(random_state=42).cost_complexity_pruning_path(X_train, y_train)
ccp_alphas, impurities = path.ccp_alphas, path.impurities

# Train multiple trees for each alpha
trees = []
for ccp_alpha in ccp_alphas:
    clf = DecisionTreeClassifier(random_state=42, ccp_alpha=ccp_alpha)
    clf.fit(X_train, y_train)
    trees.append(clf)

# Evaluate on validation data
train_scores = [clf.score(X_train, y_train) for clf in trees]
test_scores = [clf.score(X_test, y_test) for clf in trees]

# --- Plot Accuracy vs Alpha ---
plt.figure(figsize=(10, 6))
plt.plot(ccp_alphas, train_scores, marker='o', label='Train Accuracy', drawstyle="steps-post")
plt.plot(ccp_alphas, test_scores, marker='o', label='Test Accuracy', drawstyle="steps-post")
plt.xlabel("ccp_alpha")
plt.ylabel("Accuracy")
plt.title("Accuracy vs CCP Alpha (Post-Pruning)")
plt.legend()
plt.show()

# Choose best alpha (based on test accuracy)
best_alpha_index = np.argmax(test_scores)
best_tree = trees[best_alpha_index]
print(f"\n--- Post-Pruned Model Performance ---")
print(f"Best ccp_alpha: {ccp_alphas[best_alpha_index]:.5f}")
print(f"Accuracy: {test_scores[best_alpha_index] * 100:.2f}%")
print("\nClassification Report:")
y_pred_post = best_tree.predict(X_test)
print(classification_report(y_test, y_pred_post))

# --- Plot Post-Pruned Tree ---
plt.figure(figsize=(14, 7))
plot_tree(best_tree, feature_names=X.columns, class_names=["No Disease", "Has Disease"], filled=True)
plt.title("Post-Pruned Decision Tree")
plt.show()
'''