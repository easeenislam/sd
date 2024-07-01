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
    st.title("Model Inference with Two Input Tensors")

    uploaded_file1 = st.file_uploader("Choose image 1...", type=["jpg", "jpeg", "png"])
    uploaded_file2 = st.file_uploader("Choose image 2...", type=["jpg", "jpeg", "png"])

    if uploaded_file1 is not None and uploaded_file2 is not None:
        image1 = Image.open(uploaded_file1).convert('RGB')
        image1 = image1.resize((224, 224))  # Resize to match models' expected input size
        image_array1 = np.array(image1)
        image_array1 = image_array1.astype('float32') / 255.0  # Normalize image
        image_array1 = np.expand_dims(image_array1, axis=0)  # Add batch dimension

        image2 = Image.open(uploaded_file2).convert('RGB')
        image2 = image2.resize((224, 224))  # Resize to match models' expected input size
        image_array2 = np.array(image2)
        image_array2 = image_array2.astype('float32') / 255.0  # Normalize image
        image_array2 = np.expand_dims(image_array2, axis=0)  # Add batch dimension

        st.image(image1, caption='Uploaded Image 1', use_column_width=True)
        st.image(image2, caption='Uploaded Image 2', use_column_width=True)

        if st.button('Classify'):
            st.write("Classifying...")

            try:
                for model_name, model in models.items():
                    if model is None:
                        st.error(f"Model {model_name} is not loaded correctly.")
                        continue

                    st.write(f"Model: {model_name}")

                    # Assuming two input tensors are needed
                    input_tensor1 = image_array1
                    input_tensor2 = image_array2

                    # Predict using the model
                    prediction = model.predict([input_tensor1, input_tensor2])
                    pred_class = np.argmax(prediction)
                    confidence = prediction[0][pred_class]

                    st.write(f"Predicted Class: {pred_class}")
                    st.write(f"Confidence: {confidence:.2f}")

            except Exception as e:
                st.error(f"Error during prediction: {str(e)}")

if __name__ == "__main__":
    main()
