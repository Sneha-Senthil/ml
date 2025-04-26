# Step 1: Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Step 2: Load Dataset
df = pd.read_csv(r"C:\Users\sneha\Downloads\ML\ML\datasets\spam_ham_dataset (1).csv")

# Step 3: EDA
print("\n--- Data Info ---")
print(df.info())

print("\n--- First 5 Rows ---")
print(df.head())

print("\n--- Class Distribution ---")
print(df['label'].value_counts())

print("\n--- Missing Values ---")
print(df.isna().sum())

# Step 4: Preprocessing
# Encode target: ham → 0, spam → 1
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Feature (text message) and Target
X = df['text']
y = df['label']

# Step 5: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: TF-IDF Vectorization
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.95)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Step 7: Model Training
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# Step 8: Prediction
y_pred = model.predict(X_test_tfidf)

# Step 9: Evaluation Metrics
accuracy = accuracy_score(y_test, y_pred)
print(f"\n--- Model Performance ---")
print(f"Accuracy: {accuracy * 100:.2f}%")

# Step 10: Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', xticklabels=["Ham", "Spam"], yticklabels=["Ham", "Spam"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Naive Bayes Spam Detection")
plt.show()

# Step 11: Classification Report
print("\n--- Classification Report ---")
print(classification_report(y_test, y_pred, target_names=["Ham", "Spam"]))

# Step 12: Class Distribution Plot
ham_count = sum(y_test == 0)
spam_count = sum(y_test == 1)
plt.bar(["Ham", "Spam"], [ham_count, spam_count], color=['skyblue', 'salmon'])
plt.title("Ham vs Spam Count in Test Set")
plt.ylabel("Count")
plt.show()

# Step 13: Actual vs Predicted Plot
plt.figure(figsize=(6, 4))
plt.scatter(y_test, y_pred, alpha=0.5, color='purple')
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Actual vs Predicted Labels")
plt.grid(True)
plt.show()

# Step 14: Residual Plot
residuals = y_test - y_pred
plt.scatter(y_pred, residuals, alpha=0.6, color='orange')
plt.axhline(y=0, color='red', linestyle='--')
plt.xlabel("Predicted Label")
plt.ylabel("Residuals")
plt.title("Residuals vs Predicted Labels - Naive Bayes")
plt.show()
