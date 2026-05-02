import os
import cv2
import gdown
import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model


# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Face Mask Detector",
    page_icon="😷",
    layout="wide"
)


# ---------------- CONSTANTS ----------------
MODEL_PATH = "mask_detector.keras"
FILE_ID = "1WQjSlvYS93qRFBAnabkGz6XDNbHYK5zT"
MODEL_URL = f"https://drive.google.com/uc?id={FILE_ID}"
CONFIDENCE = 0.5
DISPLAY_WIDTH = 420


# ---------------- PREMIUM CSS ----------------
st.markdown("""
<style>
.stApp{
    background:#030712;
    color:white;
}

.main .block-container{
    padding-top:2rem;
    max-width:1100px;
}

section[data-testid="stSidebar"]{
    background:#111827;
    border-right:1px solid #1f2937;
}

*{
    color:white !important;
}

input, textarea{
    color:white !important;
}

section[data-testid="stSidebar"] *{
    color:white !important;
}

div[role="radiogroup"] *{
    color:white !important;
}

[data-testid="stFileUploader"]{
    background:#111827;
    border:1px solid #374151;
    border-radius:18px;
    padding:20px;
}

[data-testid="stFileUploader"] *{
    color:white !important;
}

[data-testid="stBaseButton-secondary"]{
    background:#1f2937 !important;
    color:white !important;
}

[data-testid="stAlertContainer"] *{
    color:white !important;
}

.main-title{
    text-align:center;
    font-size:56px;
    font-weight:800;
    color:white !important;
    margin-bottom:10px;
}

.subtitle{
    text-align:center;
    font-size:22px;
    color:#cbd5e1 !important;
    margin-bottom:35px;
}

div.stButton > button {
    background: #dc2626 !important;
    color: white !important;
    border: none !important;
    border-radius: 14px !important;
    padding: 14px 22px !important;
    font-size: 18px !important;
    font-weight: bold !important;
    box-shadow: 0 4px 12px rgba(220,38,38,0.4);
    transition: all 0.3s ease !important;
}

div.stButton > button:hover {
    background: #ef4444 !important;
    transform: scale(1.05);
    cursor: pointer;
}

div.stButton > button:active {
    background: #b91c1c !important;
}

header{
    background:transparent !important;
}

.result-box {
    display:flex;
    justify-content:center;
    margin-top:20px;
    margin-bottom:10px;
}
</style>
""", unsafe_allow_html=True)


# ---------------- MODEL LOAD ----------------
@st.cache_resource(show_spinner="Loading AI model... Please wait ⏳")
def load_my_model():
    if not os.path.exists(MODEL_PATH):
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
    return load_model(MODEL_PATH, compile=False)


model = load_my_model()


# ---------------- FACE DETECTOR ----------------
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


# ---------------- HEADER ----------------
st.markdown(
    """
    <div class="main-title">
    😷 Face Mask Detector
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="subtitle">
    Detect mask status from an uploaded image or browser camera photo
    </div>
    """,
    unsafe_allow_html=True
)


# ---------------- SIDEBAR ----------------
st.sidebar.header("⚙ Controls")

mode = st.sidebar.radio(
    "Choose Detection Mode",
    [
        "Upload Image",
        "Use Webcam"
    ]
)

st.sidebar.info(
    "For deployed apps, webcam works through your browser camera permission."
)


# ---------------- DETECTOR ----------------
def detect_mask(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face = image[y:y+h, x:x+w]

        if face.shape[0] == 0 or face.shape[1] == 0:
            continue

        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = cv2.resize(face, (224, 224), interpolation=cv2.INTER_CUBIC)
        face = face.astype("float32") / 255.0
        face = np.expand_dims(face, axis=0)

        pred = model.predict(face, verbose=0)[0][0]

        if pred < CONFIDENCE:
            label = "Mask"
            color = (0, 255, 0)
        else:
            label = "No Mask"
            color = (0, 0, 255)

        cv2.rectangle(image, (x, y), (x+w, y+h), color, 3)
        cv2.putText(
            image,
            label,
            (x, y-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            color,
            2
        )

    return image, len(faces)


# ---------------- IMAGE RESULT VIEW ----------------
def show_result(image_bgr, source_label):
    result, face_count = detect_mask(image_bgr.copy())
    result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image(result_rgb, caption=source_label, width=DISPLAY_WIDTH)

    if face_count == 0:
        st.warning("No face detected in the image.")
    else:
        st.success(f"Detection completed ✓ Faces detected: {face_count}")


# ---------------- IMAGE MODE ----------------
if mode == "Upload Image":
    st.info("📤 Upload an image")

    uploaded_file = st.file_uploader(
        "",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)

        if image is None:
            st.error("Could not read the uploaded image.")
        else:
            with st.spinner("Analyzing uploaded image..."):
                show_result(image, "Uploaded Image Result")


# ---------------- WEBCAM MODE ----------------
if mode == "Use Webcam":
    st.info("📷 Allow browser camera access and take a photo")

    camera_photo = st.camera_input("Open webcam")

    if camera_photo is not None:
        file_bytes = np.asarray(bytearray(camera_photo.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)

        if image is None:
            st.error("Could not read the camera image.")
        else:
            with st.spinner("Analyzing webcam photo..."):
                show_result(image, "Webcam Capture Result")