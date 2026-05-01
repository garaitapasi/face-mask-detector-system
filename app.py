import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os


# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Face Mask Detector",
    page_icon="😷",
    layout="wide"
)


# ---------------- PREMIUM CSS ----------------
st.markdown("""
<style>

/* Main app */
.stApp{
    background:#030712;
    color:white;
}


/* Main content */
.main .block-container{
    padding-top:2rem;
}


/* Sidebar */
section[data-testid="stSidebar"]{
    background:#111827;
    border-right:1px solid #1f2937;
}


/* FORCE ALL TEXT WHITE */
*{
    color:white !important;
}


/* Inputs */
input, textarea{
    color:white !important;
}


/* Sidebar */
section[data-testid="stSidebar"] *{
    color:white !important;
}


/* Radio buttons */
div[role="radiogroup"] *{
    color:white !important;
}


/* Upload widget */
[data-testid="stFileUploader"]{
    background:#111827;
    border:1px solid #374151;
    border-radius:18px;
    padding:20px;
}


/* Upload widget text */
[data-testid="stFileUploader"] *{
    color:white !important;
}


/* Upload button */
[data-testid="stBaseButton-secondary"]{
    background:#1f2937 !important;
    color:white !important;
}


/* Alerts */
[data-testid="stAlertContainer"] *{
    color:white !important;
}


/* Title */
.main-title{
    text-align:center;
    font-size:56px;
    font-weight:800;
    color:white !important;
    margin-bottom:10px;
}


/* Subtitle */
.subtitle{
    text-align:center;
    font-size:22px;
    color:#cbd5e1 !important;
    margin-bottom:35px;
}


/* ALL buttons */
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


/* Hover */
div.stButton > button:hover {

    background: #ef4444 !important;

    transform: scale(1.05);

    cursor: pointer;
}


/* Click */
div.stButton > button:active {

    background: #b91c1c !important;
}



/* Remove white header */
header{
    background:transparent !important;
}

</style>
""", unsafe_allow_html=True)


# ---------------- DOWNLOAD MODEL ----------------
if not os.path.exists("mask_detector.keras"):

    import gdown

    url = "https://drive.google.com/uc?id=1WQjSlvYS93qRFBAnabkGz6XDNbHYK5zT"

    gdown.download(
        url,
        "mask_detector.keras",
        quiet=False
    )


# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_my_model():
    if not os.path.exists("mask_detector.keras"):
        import gdown
        gdown.download(
            "https://drive.google.com/uc?id=YOUR_FILE_ID",
            "mask_detector.keras",
            quiet=False
        )
    return load_model("mask_detector.keras")

model = load_my_model()

# ---------------- FACE DETECTOR ----------------
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)


CONFIDENCE = 0.5


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
    Real-time mask detection using Deep Learning + Computer Vision
    </div>
    """,
    unsafe_allow_html=True
)


# ---------------- SIDEBAR ----------------
st.sidebar.header(
    "⚙ Controls"
)


mode = st.sidebar.radio(
    "Choose Detection Mode",
    [
        "Upload Image",
        "Live Webcam"
    ]
)


# ---------------- DETECTOR ----------------
def detect_mask(image):

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    faces = face_cascade.detectMultiScale(
        gray,
        1.3,
        5
    )

    for (x, y, w, h) in faces:

        face = image[
               y:y+h,
               x:x+w
               ]

        if face.shape[0] == 0:
            continue

        face = cv2.cvtColor(
            face,
            cv2.COLOR_BGR2RGB
        )

        face = cv2.resize(
            face,
            (224, 224),
            interpolation=cv2.INTER_CUBIC
        )

        face = face / 255.0

        face = np.expand_dims(
            face,
            axis=0
        )

        pred = model.predict(
            face,
            verbose=0
        )[0][0]

        if pred < CONFIDENCE:

            label = "Mask"
            color = (0, 255, 0)

        else:

            label = "No Mask"
            color = (0, 0, 255)

        cv2.rectangle(
            image,
            (x, y),
            (x+w, y+h),
            color,
            3
        )

        cv2.putText(
            image,
            label,
            (x, y-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            color,
            2
        )

    return image


# ---------------- IMAGE MODE ----------------
if mode == "Upload Image":

    st.info(
        "📤 Upload an image"
    )

    uploaded_file = st.file_uploader(
        "",
        type=[
            "jpg",
            "jpeg",
            "png"
        ]
    )

    if uploaded_file:

        file_bytes = np.asarray(
            bytearray(
                uploaded_file.read()
            ),
            dtype=np.uint8
        )

        image = cv2.imdecode(
            file_bytes,
            1
        )

        with st.spinner(
            "Analyzing..."
        ):

            result = detect_mask(
                image
            )

        result = cv2.cvtColor(
            result,
            cv2.COLOR_BGR2RGB
        )

        st.image(
            result,
            width=500
        )

        st.success(
            "Detection completed ✓"
        )


# ---------------- WEBCAM ----------------
if mode == "Live Webcam":

    if "camera_on" not in st.session_state:

        st.session_state.camera_on = False

    col1, col2, col3, col4 = st.columns(
        [2, 1, 1, 2]
    )

    with col2:

        if st.button(
                "▶ Start Camera",
                key="start_camera_btn"
        ):
            st.session_state.camera_on = True

    with col3:

        if st.button(
                "⏹ Stop Camera",
                key="stop_camera_btn"
        ):
            st.session_state.camera_on = False



    frame_window = st.empty()

    if st.session_state.camera_on:

        cap = cv2.VideoCapture(
            0
        )

        cap.set(
            cv2.CAP_PROP_FRAME_WIDTH,
            1280
        )

        cap.set(
            cv2.CAP_PROP_FRAME_HEIGHT,
            720
        )

        while st.session_state.camera_on:

            ret, frame = cap.read()

            if not ret:
                break

            frame = detect_mask(
                frame
            )

            frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            frame_window.image(
                frame,
                use_container_width=True
            )

        cap.release()