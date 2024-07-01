import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Function to load model with custom objects if necessary
@st.cache(allow_output_mutation=True)
def load_model(model_path):
    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model {model_path}: {e}")
        return None

# Define model paths
model_paths = {
    "DenseNet201": "DenseNet201-HPT.keras",
    "DenseNet169": "DenseNet169-HPT.keras",
    "ResNet50V2": "ResNet50V2-HPT.keras",
    "Xception": "Xception-HPT.keras",
}

# Load all models
models = {name: load_model(path) for name, path in model_paths.items() if path}

# Main application function
def main():
    st.title("Melanoma Malignant and Benign Classification App")
    st.write("Upload an image and select a model. The selected model will predict the class.")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        image = image.resize((224, 224))  # Resize to match models' expected input size

        st.image(image, caption='Uploaded Image', use_column_width=True)
        st.write("Classifying...")

        try:
            for model_name, model in models.items():
                st.write(f"Model: {model_name}")
                
                # Preprocess the image
                image_array = np.array(image)
                image_array = image_array.astype('float32') / 255.0  # Normalize image
                image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
                
                # Handle models that expect two inputs
                if model_name == "Xception":
                    input_tensor1 = image_array
                    input_tensor2 = np.zeros_like(input_tensor1)  # Example placeholder for a second input
                    prediction = model.predict([input_tensor1, input_tensor2])
                else:
                    prediction = model.predict(image_array)

                pred_class = np.argmax(prediction)
                confidence = prediction[0][pred_class]

                st.write(f"Predicted Class: {pred_class}")
                st.write(f"Confidence: {confidence:.2f}")

        except Exception as e:
            st.error(f"Error during prediction: {str(e)}")

if __name__ == "__main__":
    main()
