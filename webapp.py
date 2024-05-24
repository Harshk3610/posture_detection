

import streamlit as st
import cv2
import numpy as np
from human_posture_analysis_video import process_frame  # assuming your code is in posture_detection.py

# Title
st.title('Sitting Posture Detection')

# Permission request and description
st.markdown("""
    **This app will request access to your webcam for posture detection.**

    This deployed model uses the live feed from the camera to detect the sitting posture.
""")

# Button to start the camera
if st.button('Use Camera'):
    
    
    frameST = st.empty()

    # Start webcam
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Get fps
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0:
            fps = 30  # Assume a standard fps if actual fps cannot be determined

        # Process the frame
        output_frame = process_frame(frame, fps)

        # Display the frame
        frameST.image(output_frame, channels='BGR')

    cap.release()
