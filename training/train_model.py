# training/train_model.py
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

data = np.load("training/dataset.npz")
X, y = data['X'], data['y']

split = int(len(X) * 0.8)
X_train, X_val = X[:split], X[split:]
y_train, y_val = y[:split], y[split:]

model = keras.Sequential([
    layers.Input(shape=(6,)),
    layers.Dense(32, activation='relu'),
    layers.Dense(16, activation='relu'),
    layers.Dense(3, activation='softmax')
])

model.compile(optimizer=keras.optimizers.Adam(0.005), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=40, batch_size=16)

loss, acc = model.evaluate(X_val, y_val, verbose=0)
print(f">> Final Model Validation Accuracy Score: {acc*100:.2f}%")
if acc >= 0.88:
    model.save("training/models/base_model.h5")
    print(">> Target validation met. Model successfully saved.")
else:
    print("CRITICAL CHECK FAILURE: Accuracy below required 88% threshold.")
