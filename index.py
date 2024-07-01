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

# Load models
models = {name: load_model(path) for name, path in model_paths.items()}

# Main application function
def main():
    st.title("Melanoma Malignant and Benign Classification App")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        image = image.resize((224, 224))  # Resize to match model's expected input size
        image_array = np.array(image)
        image_array = image_array.astype('float32') / 255.0  # Normalize image
        image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension

        for model_name, model in models.items():
            st.subheader(f"Model: {model_name}")
            if model is not None:
                try:
                    # Predict class probabilities
                    prediction = model.predict(image_array)
                    pred_class = np.argmax(prediction)
                    confidence = prediction[0][pred_class]
                    
                    # Display results
                    st.write(f"Predicted Class: {pred_class}")
                    st.write(f"Confidence: {confidence:.2f}")
                except Exception as e:
                    st.error(f"Error during prediction with {model_name}: {str(e)}")
            else:
                st.error(f"Model {model_name} not loaded correctly.")

if __name__ == "__main__":
    main()
