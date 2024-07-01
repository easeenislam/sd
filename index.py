import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import streamlit_shadcn_ui as ui

# Function to load model with custom objects if necessary
@st.cache(allow_output_mutation=True)
def load_model(model_path):
    model = tf.keras.models.load_model(model_path)
    return model

# Load models
model_paths = {
    "DenseNet201": "DenseNet201-HPT.keras",
    "DenseNet169": "DenseNet169-HPT.keras",
    "ResNet50V2": "ResNet50V2-HPT.keras",
    "Xception": "Xception-HPT.keras",
}

models = {name: load_model(path) for name, path in model_paths.items()}

# Define class labels
class_labels = {0: "Benign", 1: "Malignant"}  # Adjust according to your dataset

# Function to preprocess the image
def preprocess_image(image):
    input_shape = (224, 224)  # Assuming input shape expected by models
    image = image.resize(input_shape)
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# Function to predict with the model
def predict_with_model(model, image):
    prediction = model.predict(image)
    return prediction

# Main application function
def main():
    st.sidebar.title("Dashboard")

    page = st.sidebar.radio("Go to", ["Home", "About Us"])

    if page == "Home":
        st.title("Melanoma Malignant and Benign Classification App")
        st.write("Upload an image and click Classify to predict the class.")
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, caption='Uploaded Image', use_column_width=True)
            st.write("Classifying...")

            # Preprocess the image
            processed_image = preprocess_image(image)

            classify_button = st.button("Classify")

            if classify_button:
                # Get the selected model
                selected_model = st.selectbox("Select Model", list(models.keys()))

                # Predict class probabilities using the selected model
                prediction = predict_with_model(models[selected_model], processed_image)
                
                # Get the predicted class label
                pred_class = np.argmax(prediction)

                # Map predicted class label to class name
                predicted_label = class_labels[pred_class]

                # Get the probability of the predicted class
                confidence = prediction[0][pred_class]

                st.write(f"Predicted Class: {predicted_label}")
                st.write(f"Confidence: {confidence:.2f}")

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
