import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import streamlit_shadcn_ui as ui

# Function to load model with custom objects if necessary
@st.cache(allow_output_mutation=True)
def load_model(model_path):
    try:
        model = tf.keras.models.load_model(model_path)
        model.summary()  # Print the model summary to check the input and output layers
        return model
    except Exception as e:
        st.error(f"Error loading model {model_path}: {e}")
        return None

# Load the model
model_paths = {
    "DenseNet201": "DenseNet201-HPT.keras",
    "DenseNet169": "DenseNet169-HPT.keras",
    "ResNet50V2": "ResNet50V2-HPT.keras",
    "Xception": "Xception-HPT.keras",
}  # Adjust the path accordingly
model = load_model(model_path)

# Define class labels
class_labels = {0: "Benign", 1: "Malignant"}  # Adjust according to your dataset

# Main application function
def main():
    st.sidebar.title("Dashboard")

    page = st.sidebar.radio("Go to", ["Home", "About Us"])

    if page == "Home":
        st.title("Melanoma Malignant and Benign Classification App")
        st.write("Upload an image to classify whether it is Benign or Malignant.")

        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Image', use_column_width=True)

            # Preprocess the image
            img = image.resize((224, 224))  # Resize image to match model's expected sizing
            img_array = np.array(img) / 255.0  # Normalize image
            img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

            # Duplicate the image array to match the model's input requirements
            input_1 = img_array
            input_2 = img_array

            # Predict
            try:
                prediction = model.predict([input_1, input_2])
                pred_class = np.argmax(prediction)
                st.write(f"Predicted Class: {class_labels[pred_class]}")
                st.write(f"Confidence: {prediction[0][pred_class]:.2f}")
            except Exception as e:
                st.error(f"Error predicting: {e}")

    elif page == "About Us":
        st.title("About Us")

        st.image("fn.jpg", caption="Faria Nishat")
        ui.metric_card("Faria Nishat", "Lecturer")

        cols = st.columns(2)
        with cols[0]:
            st.image("https://scontent.fdac138-1.fna.fbcdn.net/v/t39.30808-6/441458877_122128062392284217_1286163050482529306_n.jpg?_nc_cat=111&ccb=1-7&_nc_sid=5f2048&_nc_eui2=AeGEvAzVWiP52wY80pWZJDuAVuMlvLOiZIBW4yW8s6JkgD_pDT8AogdsCnUSKViE7HfPKXMervGVfDwaH3oiZ8tU&_nc_ohc=VDiUlWOxbkUQ7kNvgFRd_3x&_nc_ht=scontent.fdac138-1.fna&oh=00_AYD2s93GmOmXFT0OzgKvAJ5kML79206g0rEehE1WUZ4duA&oe=6656A719", caption="MD LIKHON MIA")
            ui.metric_card("MD LIKHON MIA", "203-15-3916", "57_D, CSE, likhon15-3916@diu.edu.bd")

        with cols[1]:
            st.image("https://scontent.fdac138-2.fna.fbcdn.net/v/t39.30808-6/440415142_977505897445068_8368649292102824103_n.jpg?_nc_cat=107&ccb=1-7&_nc_sid=5f2048&_nc_eui2=AeFZIwfZWKn87OuNsRLGLVN76U69LgfiKAXpTr0uB-IoBSK9Guqdltv84kTZTV2DE58jQVDojarvsT__ZSJPvl0j&_nc_ohc=9Pv1NzBZicAQ7kNvgHYDxF9&_nc_ht=scontent.fdac138-2.fna&oh=00_AYDYnB8k4OSO16lTTER5b1B86ChixKDpUg0ky6sQEAQTLQ&oe=6656AF8D", caption="Eshita Akter")
            ui.metric_card("Eshita Akter", "203-15-3922", "57_D, CSE, eshita15-3922@diu.edu.bd")
        
        # You can add more content about your team, project, etc.

if __name__ == "__main__":
    main()
