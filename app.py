import cv2
import streamlit as st
import numpy as np
from PIL import Image

def detect_faces():
    faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    cap = cv2.VideoCapture(0)

    stop_stream = False
    while not stop_stream:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.3,
            minNeighbors=5,
            minSize=(30, 30)
        )
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        cv2.imshow('Face Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            stop_stream = True

    cap.release()
    cv2.destroyAllWindows()
    return stop_stream


def detect_faces_in_image(uploaded_image):
    img_array = np.array(Image.open(uploaded_image))

    faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5,
        minSize=(30, 30)
    )

    for (x, y, w, h) in faces:
        cv2.rectangle(img_array, (x, y), (x + w, y + h), (0, 255, 0), 2)

    st.image(img_array, channels="BGR", use_column_width=True)

st.title("Face Detection")
st.subheader("Either Open Camera And Detect Faces Or Upload An Image And Detect Faces ")

if st.button("Open Camera"):
    if detect_faces():
        st.write("Camera stream stopped.")

uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if uploaded_image is not None:
    detect_faces_in_image(uploaded_image)
