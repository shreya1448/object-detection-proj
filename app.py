import streamlit as st
import cv2
import numpy as np
from detector import detect_objects
import os

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(page_title="Object Detection Dashboard", layout="wide")

st.title("🚀 Real-Time Object Detection Dashboard")

# -------------------------------
# Detect if running on cloud
# -------------------------------
is_cloud = os.getenv("STREAMLIT_SERVER_HEADLESS", False)

# -------------------------------
# Sidebar
# -------------------------------
st.sidebar.header("Settings")

if not is_cloud:
    run_webcam = st.sidebar.checkbox("Start Webcam")
else:
    st.sidebar.warning("⚠️ Webcam not supported on deployed app")
    run_webcam = False

uploaded_file = st.sidebar.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

# -------------------------------
# Webcam Section (LOCAL ONLY)
# -------------------------------
if run_webcam:
    cap = cv2.VideoCapture(0)
    stframe = st.empty()

    if not cap.isOpened():
        st.error("Failed to access webcam")
    else:
        while run_webcam:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to read frame")
                break

            annotated_frame = detect_objects(frame)
            stframe.image(annotated_frame, channels="BGR")

        cap.release()

# -------------------------------
# Image Upload Section
# -------------------------------
if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    st.subheader("📷 Uploaded Image")
    st.image(image, channels="BGR")

    # Run detection
    result = detect_objects(image)

    st.subheader("🎯 Detection Result")
    st.image(result, channels="BGR")

    # Download button
    _, buffer = cv2.imencode(".jpg", result)
    st.download_button(
        label="Download Result",
        data=buffer.tobytes(),
        file_name="detected.jpg",
        mime="image/jpeg"
    )

# -------------------------------
# Footer
# -------------------------------
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit & YOLOv8")