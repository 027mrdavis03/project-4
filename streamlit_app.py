import streamlit as st
import pickle
from PIL import Image
import numpy as np

with open('hotdog_notdog_cnn_regularized_augmented_4_layers.pkl', 'rb') as f:
    model = pickle.load(f)

# Define a function to preprocess the input image
def preprocess_image(image):
    image = image.resize((299, 299))
    image = np.array(image)
    image = image / 255.0  
    image = np.expand_dims(image, axis=0)  
    return image

st.title("Hot dog or no hot dog? 🌭")
st.write(
    "Is it really a hot dog? Upload your pic to find out!"
)

# File uploader for image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    try:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_column_width=True)

        # Preprocess the image
        preprocessed_image = preprocess_image(image)

        # Make prediction
        prediction = model.predict(preprocessed_image)

        # Convert prediction to readable label
        class_labels = ["Not a Hot Dog", "Hot Dog"]  
        predicted_class = class_labels[np.argmax(prediction)]

        # Display prediction
        st.write("Prediction:", predicted_class)
    except Exception as e:
        st.error(f"An error occurred: {e}")