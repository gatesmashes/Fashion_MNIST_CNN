import streamlit as st
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image

# Load the model (choose .keras or .h5 file)
model = keras.models.load_model("model.keras")

# Fashion MNIST class labels
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

st.title("👕 Fashion MNIST Classifier")
st.write("Upload a grayscale clothing image (28x28 px) to predict the category.")

uploaded_file = st.file_uploader("Upload an image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Show uploaded image
    image = Image.open(uploaded_file).convert("L")   # convert to grayscale
    st.image(image, caption="Uploaded Image", width=150)

    # Preprocess image
    img_resized = image.resize((28, 28))
    img_array = np.array(img_resized) / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)

    # Prediction
    prediction = model.predict(img_array)
    pred_class = np.argmax(prediction)

    st.write(f"### Prediction: {class_names[pred_class]}")
    st.bar_chart(prediction[0])
