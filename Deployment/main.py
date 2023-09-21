import streamlit as st
# import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

# Load your pre-trained model
# model = tf.keras.models.load_model('your_model.h5')

# Define class labels (replace with your own)
class_labels = ["Class 1", "Class 2", "Class 3", "Class 4"]

# Streamlit UI
st.title('Waste Sorter App')

# Upload an image
uploaded_image = st.file_uploader("Upload an image...", type=["jpg", "png", "jpeg"])

if uploaded_image is not None:
    # Create two columns
    col1, col2 = st.columns(2)

    # Display the uploaded image in the first column
    with col1:
        st.image(uploaded_image, caption='Uploaded Image', use_column_width=True)

    # Perform inference in the second column
    with col2:
        # Display the class probabilities
        st.write("Class Probabilities:")
        image = Image.open(uploaded_image)
        # Preprocess the image (resize, normalize, etc.) as required by your model
        # Make predictions using your model
        prediction = model.predict(np.array(image).reshape(1, image.size[1], image.size[0], 3))

        # Create a bar plot for class probabilities
        fig, ax = plt.subplots()
        ax.bar(class_labels, prediction[0])
        ax.set_xlabel('Class Labels')
        ax.set_ylabel('Probability')
        ax.set_title('Class Probabilities')
        st.pyplot(fig)

# Run the Streamlit app
if __name__ == '__main__':
    st.write("To use this app, please upload an image.")
