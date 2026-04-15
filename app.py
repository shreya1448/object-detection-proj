import streamlit as st
import cv2
import numpy as np
import time
from detector import detect_objects

st.set_page_config(page_title="Object Detection Dashboard", layout="wide")

st.title("🚀 Real-Time Object Detection Dashboard")

# Sidebar
st.sidebar.header("Settings")
run_webcam = st.sidebar.checkbox("Start Webcam")
capture_frame = st.sidebar.button("📸 Capture Frame")

# Layout
col1, col2 = st.columns(2)

frame_placeholder = col1.empty()
info_placeholder = col2.empty()

# =========================
# 🔹 WEBCAM MODE
# =========================
if run_webcam:
    cap = cv2.VideoCapture(0)
    prev_time = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to access webcam")
            break

        frame = cv2.resize(frame, (640, 480))

        # FPS calculation
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
        prev_time = curr_time

        annotated_frame, labels, counts = detect_objects(frame)

        # Add FPS text
        cv2.putText(annotated_frame, f"FPS: {int(fps)}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 2)

        # Convert for Streamlit
        display_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

        frame_placeholder.image(display_frame, channels="RGB")

        # Info panel
        with info_placeholder.container():
            st.subheader("Detected Objects")
            for label, conf in labels:
                st.write(f"{label} ({conf:.2f})")

            st.subheader("Object Count")
            st.write(counts)

            # Alerts
            if "person" in counts:
                st.warning("⚠️ Person detected!")

        # Capture + Download
        if capture_frame:
            _, buffer = cv2.imencode('.jpg', annotated_frame)
            st.download_button(
                label="Download Captured Frame",
                data=buffer.tobytes(),
                file_name="captured.jpg",
                mime="image/jpeg"
            )

# =========================
# 🔹 IMAGE UPLOAD MODE
# =========================
st.sidebar.header("Upload Image")
uploaded_file = st.sidebar.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    frame = cv2.imdecode(file_bytes, 1)

    annotated_frame, labels, counts = detect_objects(frame)

    # Convert for display
    display_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

    col1.image(display_frame, channels="RGB")

    with col2:
        st.subheader("Detected Objects")
        for label, conf in labels:
            st.write(f"{label} ({conf:.2f})")

        st.subheader("Object Count")
        st.write(counts)

        # Alerts
        if "person" in counts:
            st.warning("⚠️ Person detected!")

        # ✅ Download button (correct placement)
        _, buffer = cv2.imencode('.jpg', annotated_frame)
        st.download_button(
            label="Download Image",
            data=buffer.tobytes(),
            file_name="detected.jpg",
            mime="image/jpeg"
        )