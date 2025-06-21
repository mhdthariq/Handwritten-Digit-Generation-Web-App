import streamlit as st
import tensorflow as tf
from tensorflow.keras import models
import numpy as np
import matplotlib.pyplot as plt
import random

# --- App Configuration ---
st.set_page_config(layout="wide")
st.title("MNIST Digit Classifier")
st.write("Test the trained deep learning model by selecting a digit. The app will randomly select 5 images of that digit from the test set and show the model's prediction for each.")

# --- Load Model and Data ---
# Use st.cache_resource to load the model and data only once
@st.cache_resource
def load_model_and_data():
    """Loads the saved Keras model and the MNIST test dataset."""
    try:
        # Load the entire model
        model = models.load_model("../models/bestmodel.h5")
    except (IOError, ImportError):
        st.error("Error: Could not load 'bestmodel.h5'. Make sure it's in the 'models' directory and has been uploaded to your deployment environment.")
        return None, None, None
    
    # Load the test data
    (_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    
    # Preprocess the test data (normalization and reshaping)
    x_test_normalized = x_test.astype("float32") / 255.0
    x_test_reshaped = np.expand_dims(x_test_normalized, -1)
    
    return model, x_test_reshaped, y_test

# Load the resources
model, x_test, y_test = load_model_and_data()

# --- Web App Interface ---
if model is not None:
    st.sidebar.header("Controls")
    
    # User selects a digit from the dropdown
    selected_digit = st.sidebar.selectbox(
        "Choose a digit to test (0-9):",
        list(range(10))
    )

    # Button to trigger the prediction
    if st.sidebar.button("Show Predictions"):
        st.subheader(f"Model Predictions for Digit: {selected_digit}")
        
        # Find all indices in the test set that match the selected digit
        indices = np.where(y_test == selected_digit)[0]

        if len(indices) < 5:
            st.warning(f"Warning: Found fewer than 5 examples of the digit '{selected_digit}' to display.")
            num_images_to_show = len(indices)
        else:
            num_images_to_show = 5
            # Pick 5 unique random indices
            random_indices = random.sample(list(indices), num_images_to_show)

        # Create columns to display images side-by-side
        cols = st.columns(num_images_to_show)

        for i in range(num_images_to_show):
            with cols[i]:
                # Get the specific image and its true label
                test_image = x_test[random_indices[i]]
                true_label = y_test[random_indices[i]]

                # Prepare image for prediction (add batch dimension)
                image_for_prediction = np.expand_dims(test_image, axis=0)

                # Make a prediction
                prediction_probabilities = model.predict(image_for_prediction)
                predicted_digit = np.argmax(prediction_probabilities)

                # Display the original image (not the reshaped one)
                fig, ax = plt.subplots()
                ax.imshow(test_image.squeeze(), cmap="gray_r")
                
                # Set title color based on correctness
                title_color = "green" if predicted_digit == true_label else "red"
                ax.set_title(f"Prediction: {predicted_digit}", color=title_color)
                ax.axis("off")
                st.pyplot(fig)
else:
    st.error("Application cannot start because the model could not be loaded.")