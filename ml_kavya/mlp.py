
import os
import struct
import numpy as np
import seaborn as sns
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# Load MNIST data
def load_mnist_images(filename):
    with open(filename, 'rb') as f:
        magic, num, rows, cols = struct.unpack('>IIII', f.read(16))
        images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, rows * cols)
        return images.astype('float32') / 255.0

def load_mnist_labels(filename):
    with open(filename, 'rb') as f:
        magic, num = struct.unpack('>II', f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)
        return labels
    
# Set data path
data_path = "mnist"

x_train = load_mnist_images(os.path.join(data_path, 'train-images.idx3-ubyte'))
y_train = load_mnist_labels(os.path.join(data_path, 'train-labels.idx1-ubyte'))
x_test = load_mnist_images(os.path.join(data_path, 't10k-images.idx3-ubyte'))
y_test = load_mnist_labels(os.path.join(data_path, 't10k-labels.idx1-ubyte'))

# One-hot encode labels
y_train_oh = to_categorical(y_train, 10)
y_test_oh = to_categorical(y_test, 10)

# Build the MLP model
model = models.Sequential([
    layers.Dense(128, activation='relu', input_shape=(784,)),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(x_train, y_train_oh, epochs=50, batch_size=64, validation_split=0.1)

# Evaluate on test set
loss, acc = model.evaluate(x_test, y_test_oh)
print(f'Test Accuracy: {acc * 100:.2f}%')

# Plot Accuracy and Loss
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title('Accuracy over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()
plt.show()

# -------- Classification Metrics --------
# Predict
y_pred_probs = model.predict(x_test)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = y_test  # already integer labels, not one-hot

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu',
            xticklabels=[f'Pred {i}' for i in range(10)],
            yticklabels=[f'True {i}' for i in range(10)])
plt.title('MNIST - Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
plt.show()

# Classification Report
print("Detailed Classification Report:")
print(classification_report(y_true, y_pred, digits=4))

# Overall Accuracy
print(f"Overall Test Accuracy: {accuracy_score(y_true, y_pred) * 100:.2f}%")

# Show 10 test images with predictions
plt.figure(figsize=(10, 5))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(x_test[i].reshape(28, 28), cmap='gray')
    plt.title(f"Pred: {y_pred[i]}\nTrue: {y_true[i]}")
    plt.axis('off')
plt.tight_layout()
plt.show()
