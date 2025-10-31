import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# iris_classifier.py
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# -----------------------------
# 1️⃣ Load and Prepare the Data
# -----------------------------
iris = load_iris()
X = iris.data
y = iris.target

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42)

# -----------------------------
# 2️⃣ Build the Neural Network
# -----------------------------
model = keras.Sequential([
    layers.Dense(8, activation='relu', input_shape=(4,)),   # hidden layer
    layers.Dense(3, activation='softmax')                   # output layer
])

# -----------------------------
# 3️⃣ Compile the Model
# -----------------------------
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# -----------------------------
# 4️⃣ Train the Model
# -----------------------------
history = model.fit(X_train, y_train,
                    epochs=50,
                    batch_size=8,
                    validation_split=0.2,
                    verbose=1)

# -----------------------------
# 5️⃣ Evaluate on Test Data
# -----------------------------
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"\nTest Accuracy: {test_acc:.3f}")
print(f"Test Loss: {test_loss:.3f}")

# -----------------------------
# 6️⃣ Visualize Learning Curves
# -----------------------------
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.plot(history.history['loss'], label='Training loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss over Epochs')
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history['accuracy'], label='Training accuracy')
plt.plot(history.history['val_accuracy'], label='Validation accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy over Epochs')
plt.legend()

plt.show()

# -----------------------------
# 7️⃣ Predict and Visualize Results
# -----------------------------
predictions = model.predict(X_test)
predicted_classes = np.argmax(predictions, axis=1)

cm = confusion_matrix(y_test, predicted_classes)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=iris.target_names)
disp.plot(cmap=plt.cm.Purples)
plt.title("Iris Classification Confusion Matrix")
plt.show()
