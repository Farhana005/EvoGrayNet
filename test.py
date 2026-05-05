import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np

model = build_model(random_architecture)  # Pass the appropriate architecture to `build_model`


# Path to the saved model weights
model_path = '/kaggle/working/best_model_kvasir.h5'

# Load the model weights
model.load_weights(model_path)

# Optionally, compile the model if you need to compute metrics
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Evaluate the model on the test dataset
test_loss, test_accuracy = model.evaluate(x_test, y_test, batch_size=1, verbose=1)

# Print the results
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")


