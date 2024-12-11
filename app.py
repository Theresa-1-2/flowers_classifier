# import streamlit as st
# import pickle
# from tensorflow.keras.preprocessing.image import load_img, img_to_array
# import numpy as np
# from PIL import Image

# # Path to the trained model
# MODEL_PATH = "model.pkl"  # Replace with the actual path to your saved .pkl model

# # Class names for the flower dataset
# CLASS_NAMES = ['Daisy', 'Tulip', 'Sunflower', 'Rose', 'Dandelion']  # Replace with your actual class names

# # Function to preprocess the uploaded image
# def preprocess_image(image):
#     img = image.resize((128, 128))  # Resize image to match the model's input size
#     img_array = img_to_array(img) / 255.0  # Normalize pixel values to [0, 1]
#     img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
#     return img_array

# # Load the trained model
# @st.cache_resource  # Cache the model to avoid reloading
# def load_trained_model():
#     with open(MODEL_PATH, 'rb') as f:
#         model = pickle.load(f)
#     return model

# model = load_trained_model()

# # Streamlit app interface
# st.title("Flower Classification App")
# st.write("Upload an image of a flower to predict its type!")

# # File uploader
# uploaded_file = st.file_uploader("Upload an image file", type=["jpg", "jpeg", "png"])

# if uploaded_file is not None:
#     # Display the uploaded image
#     image = Image.open(uploaded_file)
#     st.image(image, caption="Uploaded Image", use_column_width=True)
    
#     # Preprocess the image
#     processed_image = preprocess_image(image)
    
#     # Make predictions
#     predictions = model.predict(processed_image)
#     predicted_class = CLASS_NAMES[np.argmax(predictions)]
#     confidence = np.max(predictions)
    
#     # Display predictions
#     st.write(f"**Prediction:** {predicted_class}")
#     st.write(f"**Confidence:** {confidence:.2%}")

#####################################################################################################

import streamlit as st
import pickle
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
from PIL import Image

# Path to the trained model
MODEL_PATH = "flower_classification_model.pkl"  # Replace with the actual path to your saved .pkl model

# Class names for the flower dataset
CLASS_NAMES = ['Daisy', 'Tulip', 'Sunflower', 'Rose', 'Dandelion']  # Replace with your actual class names

# Function to preprocess the uploaded image
def preprocess_image(image):
    img = image.resize((150, 150))  # Resize image to match the model's input size
    img_array = img_to_array(img) / 255.0  # Normalize pixel values to [0, 1]
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Load the trained model
@st.cache_resource  # Cache the model to avoid reloading it each time
def load_trained_model():
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    return model

model = load_trained_model()

# Streamlit app interface
st.title("Flower Classification App")
st.write("Upload an image of a flower to predict its type!")

# File uploader
uploaded_file = st.file_uploader("Upload an image file", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Preprocess the image
    processed_image = preprocess_image(image)
    
    # Make predictions
    predictions = model.predict(processed_image)
    predicted_class = CLASS_NAMES[np.argmax(predictions)]
    confidence = np.max(predictions)
    
    # Display predictions
    st.write(f"**Prediction:** {predicted_class}")
    st.write(f"**Confidence:** {confidence:.2%}")
