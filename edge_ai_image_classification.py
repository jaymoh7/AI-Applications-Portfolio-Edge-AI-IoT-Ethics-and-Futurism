"""
Edge AI Image Classification using TensorFlow and TensorFlow Lite

This script trains a lightweight CNN model to classify images 
(e.g., rock, paper, scissors) and converts it to TensorFlow Lite 
for edge deployment.

Author: James Njoroge
Date: 2025-07-22
"""

import tensorflow as tf
from tensorflow.keras import layers, models
import pathlib

# Load dataset from TensorFlow's example repository
dataset_url = "https://storage.googleapis.com/download.tensorflow.org/data/rps.zip"
data_dir = tf.keras.utils.get_file(origin=dataset_url, fname='rps', extract=True)
data_dir = pathlib.Path(data_dir).parent / "rps"

# Parameters
batch_size = 32
img_height = 150
img_width = 150
epochs = 5

# Load and split dataset
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

# Define the CNN model
model = models.Sequential([
    layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
    layers.Conv2D(16, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(3, activation='softmax')  # 3 classes: rock, paper, scissors
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(train_ds, validation_data=val_ds, epochs=epochs)

# Evaluate the model
val_loss, val_acc = model.evaluate(val_ds)
print(f"Validation Accuracy: {val_acc:.2f}")

# Convert the model to TensorFlow Lite format
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the TFLite model
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)

print("TFLite model successfully saved as 'model.tflite'")
