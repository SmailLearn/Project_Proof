import streamlit as st
from PIL import Image as PilImage
import tensorflow as tf
from tensorflow.keras.preprocessing import image as tf_image
import pickle

# Load the model path
model_path = 'C:/Users/1/Desktop/MSDE5/Deep_Learning_MODULE_7/final_project/dog_cat_detector_model_Final_1.pkl'

# To load the model
with open(model_path, 'rb') as file:
    try:
        loaded_model = pickle.load(file)
    except Exception as e:
        st.error(f"Error loading the model: {e}")
        st.stop()

# Streamlit app
st.title("Cat or Dog Classifier")

# Upload image through Streamlit
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    # Display the uploaded image
    pil_image = PilImage.open(uploaded_file)
    st.image(pil_image, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image for the model
    img_array = tf_image.img_to_array(pil_image)
    img_array = tf.image.resize(img_array, (224, 224))  # Resize to match the model's input shape
    img_array = tf.expand_dims(img_array, 0)  # Create batch axis
    img_array /= 255.0  # Normalize pixel values to [0, 1]

    # Make predictions using the loaded model with modified input layer
    predictions = loaded_model.predict(img_array)

    # Print raw predictions
    st.write("Raw Predictions:", predictions[0].tolist())

    # Manually interpret predictions based on your model's output with a threshold
    threshold = 0.5
    predicted_class = 'Dog' if predictions[0] >= threshold else 'Cat'
    confidence = predictions[0]

    # Display the result
    st.subheader("Prediction:")
    st.write(f"Predicted class: {predicted_class}")
    st.write(f"Confidence: {float(confidence):.2%}")
