import tensorflow as tf
from tensorflow.keras import models
import numpy as np
import matplotlib.pyplot as plt
import random

# --- 1. Load the Trained Model and the Test Data ---

print("Loading the trained model: bestmodel.h5...")
try:
    # Load the entire model (architecture, weights, and optimizer state)
    best_model = models.load_model("bestmodel.h5")
except (IOError, ImportError) as e:
    print("\nError: Could not load 'bestmodel.h5'.")
    print("Please make sure the 'bestmodel.h5' file is in the same directory as this script.")
    exit()

print("Model loaded successfully.")

# Load the test data
(_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Preprocess the test data just like the training data
x_test_normalized = x_test.astype("float32") / 255.0
x_test_reshaped = np.expand_dims(x_test_normalized, -1)


# --- 2. Inference Loop to Test the Model ---

while True:
    # Prompt the user for input
    user_input = input("\nEnter a digit (0-9) to test, or type 'q' to quit: ")

    if user_input.lower() == 'q':
        print("Exiting.")
        break

    try:
        # Convert user input to an integer
        digit_to_find = int(user_input)
        if not 0 <= digit_to_find <= 9:
            raise ValueError

    except ValueError:
        print("Invalid input. Please enter a single digit from 0 to 9.")
        continue

    # Find all indices in the test set that match the requested digit
    indices = np.where(y_test == digit_to_find)[0]

    if len(indices) == 0:
        print(f"No images of the digit '{digit_to_find}' found in the test set.")
        continue

    # Pick a random index from the list of matching indices
    random_index = random.choice(indices)

    # Get the image and its true label
    test_image = x_test_reshaped[random_index]
    true_label = y_test[random_index]

    # The model expects a batch of images, so we add an extra dimension
    image_for_prediction = np.expand_dims(test_image, axis=0)

    # --- 3. Make a Prediction ---
    prediction_probabilities = best_model.predict(image_for_prediction)
    
    # The output is an array of probabilities. The highest one is our prediction.
    predicted_digit = np.argmax(prediction_probabilities)

    # --- 4. Display the Result ---
    # We use the original (non-normalized) image for a clearer display
    image_to_show = x_test[random_index]
    
    plt.imshow(image_to_show, cmap=plt.cm.binary)
    plt.title(f"True Label: {true_label}\nPredicted Digit: {predicted_digit}")
    plt.axis('off') # Hide the axes
    plt.show()