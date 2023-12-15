import streamlit as st
from PIL import Image as PilImage
import tensorflow as tf
from tensorflow.keras.preprocessing import image as tf_image
import pickle

# Load the model path
model_path = r'C:\Users\1\Desktop\MSDE5\Deep_Learning_MODULE_7\final_project\dog_cat_detector_model_Final.pkl'

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
    img_array = tf.image.resize(img_array, (350, 380))  # Adjust the dimensions to match the model's new input shape
    img_array = tf.expand_dims(img_array, 0)  # Create batch axis

    # Make predictions using the loaded model with modified input layer
    predictions = loaded_model.predict(img_array)

    # Print raw predictions
    st.write("Raw Predictions:", predictions[0].tolist())

    # Manually interpret predictions based on your model's output
    class_index = tf.argmax(predictions[0])
    class_names = {0: 'Cat', 1: 'Dog'}  # Adjust these labels based on your model's classes
    predicted_class = class_names[class_index.numpy()]
    confidence = tf.reduce_max(predictions[0]).numpy()

    # Display the result
    st.subheader("Prediction:")
    st.write(f"Predicted class: {predicted_class}")
    st.write(f"Confidence: {confidence:.2%}")