import streamlit as st
from PIL import Image
from utils import predict_image


# Page Configuration
st.set_page_config(
    page_title="Lung Cancer Detection",
    layout="centered"
)


# Title
st.title("Lung Cancer Detection System")

st.markdown("""
Upload a lung CT scan image and the AI model will predict:

- Begin
- Malignant
- Normal
""")


# File Upload
uploaded_file = st.file_uploader(
    "Upload Lung CT Scan Image",
    type=["jpg", "jpeg", "png"]
)


# If Image Uploaded
if uploaded_file is not None:

    # Open image using PIL
    image = Image.open(uploaded_file)

    # Display image
    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    # Predict Button
    if st.button("Predict"):

        # Loading Spinner
        with st.spinner("Analyzing image..."):

            # Prediction
            predicted_class, confidence, probabilities = predict_image(image)

        # Prediction Result
        
        st.success(f"Prediction: {predicted_class}")

        st.info(f"Confidence: {confidence:.2f}%")

        
        # Probability Scores
        
        st.subheader("Prediction Probabilities")

        for label, prob in probabilities.items():
            st.write(f"{label}: {prob:.2f}%")
            st.progress(int(prob))

