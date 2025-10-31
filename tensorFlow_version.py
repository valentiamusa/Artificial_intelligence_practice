
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# 1️⃣ Define the model structure
model = keras.Sequential([
    layers.Dense(8, activation='relu', input_shape=(4,)),  # hidden layer
    layers.Dense(3, activation='softmax')                   # output layer
])

# 2️⃣ Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 3️⃣ Display model summary
model.summary()

