import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import numpy as np
from sklearn.model_selection import train_test_split
import os

# --- Parameters ---
NUM_EPOCHS = 20 # We can set a higher max epochs, early stopping will find the best one
BATCH_SIZE = 128
NUM_CLASSES = 10
INPUT_SHAPE = (28, 28, 1)

# --- 1. Load and Prepare the MNIST Dataset ---

(x_train_full, y_train_full), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_train_full = x_train_full.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

x_train_full = np.expand_dims(x_train_full, -1)
x_test = np.expand_dims(x_test, -1)

x_train, x_val, y_train, y_val = train_test_split(
    x_train_full, y_train_full, test_size=0.2, random_state=42
)

# --- 2. Build the Convolutional Neural Network (CNN) Model ---

def build_classifier_model():
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=INPUT_SHAPE),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(NUM_CLASSES, activation='softmax')
    ])
    return model

model = build_classifier_model()

# --- 3. Compile the Model ---

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# --- NEW: Define Callbacks for Early Stopping and Best Model Saving ---

# ModelCheckpoint will save the best model based on validation loss
# The user requested the filename 'bestmodel.h5'
model_checkpoint_cb = ModelCheckpoint(
    filepath="bestmodel.h5",
    save_weights_only=False, # Save the full model
    monitor='val_loss',      # Monitor validation loss
    mode='min',              # We want to minimize the loss
    save_best_only=True      # Only save when the model is the "best"
)

# EarlyStopping will halt training if the validation loss doesn't improve
early_stopping_cb = EarlyStopping(
    monitor='val_loss', # Monitor validation loss
    patience=3,         # Stop after 3 epochs of no improvement
    restore_best_weights=True # Restore weights from the best epoch
)


# --- 4. Train the Model with Callbacks ---

print("\nStarting training with Early Stopping and Model Checkpoint...")

history = model.fit(
    x_train,
    y_train,
    batch_size=BATCH_SIZE,
    epochs=NUM_EPOCHS,
    validation_data=(x_val, y_val),
    # Pass the list of callbacks to the training process
    callbacks=[model_checkpoint_cb, early_stopping_cb]
)

print("\nTraining finished.")


# --- 5. Evaluate the Best Saved Model ---

# Load the best model that was saved by ModelCheckpoint
print("\nLoading the best model saved as 'bestmodel.h5'...")
best_model = models.load_model("bestmodel.h5")

# Evaluate its performance on the final test set
print("Evaluating the best model on the test set...")
test_loss, test_acc = best_model.evaluate(x_test, y_test, verbose=2)
print(f"\nBest Model - Test accuracy: {test_acc:.4f}")
print(f"Best Model - Test loss: {test_loss:.4f}")