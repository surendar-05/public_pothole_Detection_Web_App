import os
import cv2
import tempfile
import matplotlib.pyplot as plt
from pathlib import Path
from ultralytics import YOLO
import streamlit as st
from streamlit_option_menu import option_menu

# Set page configuration
st.set_page_config(page_title="Pothole Detection", layout="wide", page_icon="🛠️")

# Sidebar for navigation
with st.sidebar:
    selected = option_menu(
        'Pothole Detection System',
        ['Image Upload', 'Video Upload'],
        menu_icon='tools',
        icons=['image', 'film'],
        default_index=0
    )

# Define YOLO model path
MODEL_PATH = "pothole_segmentation.pt"

# Check if the model exists
if not Path(MODEL_PATH).exists():
    st.error(f"Model file '{MODEL_PATH}' not found. Please upload it to the project directory.")

# Image Upload Page
if selected == 'Image Upload':
    st.title('Pothole Detection Using Deep Learning')

    def detect_objects(image_path, model_path):
        """
        Detect objects in the input image using a YOLO model.

        Args:
            image_path (str): Path to the input image.
            model_path (str): Path to the pre-trained YOLO model.

        Returns:
            Annotated image with detected objects.
        """
        # Load the model
        model = YOLO(model_path)

        # Run detection on the image
        results = model.predict(source=image_path, conf=0.5)

        # Convert detections into an annotated image
        annotated_image = results[0].plot()

        return annotated_image

    # Input from user for image
    uploaded_image = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        # Save the uploaded image temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_image_file:
            temp_image_file.write(uploaded_image.getbuffer())
            image_path = temp_image_file.name

        # Perform object detection
        output_image = detect_objects(image_path, MODEL_PATH)

        # Convert BGR to RGB for displaying with matplotlib
        output_image = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)

        # Create two columns to display the input and output images side by side
        col1, col2 = st.columns(2)

        with col1:
            st.image(uploaded_image, caption="Input Image", use_container_width=True)

        with col2:
            st.image(output_image, caption="Detected Pothole", use_container_width=True)

# Video Upload Page
if selected == 'Video Upload':
    st.title('Pothole Detection for Video')

    def detect_potholes_yolov8(frame, model):
        """
        Detect potholes in the input frame using YOLO.

        Args:
            frame (ndarray): Input video frame.
            model: Loaded YOLO model.

        Returns:
            Processed frame with bounding boxes and labels.
        """
        # Run detection on the frame
        results = model.predict(frame, conf=0.5)

        # Draw bounding boxes and labels
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()  # Bounding box coordinates
            confidences = result.boxes.conf.cpu().numpy()  # Confidence scores
            class_ids = result.boxes.cls.cpu().numpy()  # Class IDs

            for box, confidence, class_id in zip(boxes, confidences, class_ids):
                x1, y1, x2, y2 = map(int, box)
                label = f"Pothole {confidence:.2f}"

                # Draw rectangle and label
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 255, 0), 2)

        return frame

    # Load the YOLO model
    model = YOLO(MODEL_PATH)

    # Video file uploader
    video_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])

    if video_file is not None:
        # Save the uploaded video temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video_file:
            temp_video_file.write(video_file.getbuffer())
            video_path = temp_video_file.name

        # Open the video file
        cap = cv2.VideoCapture(video_path)

        # Create columns to display both videos side by side
        col1, col2 = st.columns(2)
        col1.header("Input Video")
        col2.header("Detected Potholes")

        # Create placeholders for the videos
        input_video_frame = col1.empty()
        output_video_frame = col2.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Show the original frame in the input video column
            input_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            input_video_frame.image(input_frame, channels="RGB", use_container_width=True)

            # Detect potholes in the current frame
            processed_frame = detect_potholes_yolov8(frame, model)

            # Convert BGR to RGB for Streamlit
            processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)

            # Show the processed frame in the output video column
            output_video_frame.image(processed_frame, channels="RGB", use_container_width=True)

        cap.release()
        cv2.destroyAllWindows()
