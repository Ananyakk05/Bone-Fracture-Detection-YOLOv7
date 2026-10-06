# import os
# import cv2
# import numpy as np
# import streamlit as st
# import onnxruntime as ort
# from matplotlib.colors import TABLEAU_COLORS 
# from pathlib import Path


# ALLOWED_EXTENSIONS = {"txt", "pdf", "png", "jpg", "jpeg", "gif"}
# parent_root = Path(__file__).parent.parent.absolute().__str__() # os.path.dirname(os.path.abspath(__file__))
# h, w = 640, 640
# model_onnx_path = os.path.join(parent_root, "yolov7-p6-bonefracture.onnx")
# device = "cuda"

# def color_list():
#     # Return first 10 plt colors as (r,g,b) https://stackoverflow.com/questions/51350872/python-from-color-name-to-rgb
#     def hex2rgb(h):
#         return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))

#     return [hex2rgb(h) for h in TABLEAU_COLORS.values()]

# colors = color_list()

# def xyxy2xywhn(bbox, H, W):

#     x1, y1, x2, y2 = bbox

#     return [0.5*(x1+x2)/W, 0.5*(y1+y2)/H, (x2-x1)/W, (y2-y1)/H]

# def xywhn2xyxy(bbox, H, W):

#     x, y, w, h = bbox

#     return [(x-w/2)*W, (y-h/2)*H, (x+w/2)*W, (y+h/2)*H]

# def load_img(uploaded_file):
#     """ Load image from bytes to numpy
#     """

#     file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
#     opencv_image = cv2.imdecode(file_bytes, 1)
#     return opencv_image[..., ::-1]

# def preproc(img):
#     """ Image preprocessing
#     """
#     img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
#     img = img.astype(np.float32).transpose(2, 0, 1)/255
#     return np.expand_dims(img, axis=0)

# def model_inference(model_path, image_np, device="cpu"):

#     providers = ["CUDAExecutionProvider"] if device=="cuda" else ["CPUExecutionProvider"]
#     session = ort.InferenceSession(model_path, providers=providers)
#     input_name = session.get_inputs()[0].name
#     output_name = session.get_outputs()[0].name
#     output = session.run([output_name], {input_name: image_np})

#     return output[0][:, :6]


# def post_process(img, output, score_threshold=0.3):
#     """
#     Draw bounding boxes on the input image. Dump boxes in a txt file.
#     """
#     # assert format == "xyxy" or format == "xywh"

#     det_bboxes, det_scores, det_labels = output[:, 0:4], output[:, 4], output[:, 5]
#     id2names = {
#         0: "boneanomaly", 1: "bonelesion", 2: "foreignbody", 
#         3: "fracture", 4: "metal", 5: "periostealreaction", 
#         6: "pronatorsign", 7:"softtissue", 8:"text"
#     }

#     if isinstance(img, str):
#         img = cv2.imread(img)
    
#     img = img.astype(np.uint8)
#     H, W = img.shape[:2]
#     label_txt = ""

#     for idx in range(len(det_bboxes)):
#         if det_scores[idx]>score_threshold:
#             bbox = det_bboxes[idx]
#             label = det_labels[idx]
            
#             bbox = xyxy2xywhn(bbox, h, w)
#             label_txt += f"{int(label)} {det_scores[idx]:.5f} {bbox[0]:.5f} {bbox[1]:.5f} {bbox[2]:.5f} {bbox[3]:.5f}\n"

#             bbox = xywhn2xyxy(bbox, H, W)
#             bbox_int = [int(x) for x in bbox]
#             x1, y1, x2, y2 = bbox_int
#             color_map = colors[int(label)]
#             txt = f"{id2names[label]} {det_scores[idx]:.2f}"
#             (text_width, text_height), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            
#             cv2.rectangle(img, (x1, y1), (x2, y2), color_map, 2)
#             cv2.rectangle(img, (x1-2, y1-text_height-10), (x1 + text_width+2, y1), color_map, -1)
#             cv2.putText(img, txt, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
#     return img, label_txt


# if __name__ == "__main__":

#     st.title("Bone Fracture Detection")
    
#     uploaded_file = st.file_uploader("Choose a image file", type=["png", "jpg", "jpeg", "gif"])
    
#     if uploaded_file is not None:
        
#         conf_thres = st.slider("Object confidence threshold", 0.2, 1., step=0.05)

#         # load and display orignal image
#         img = load_img(uploaded_file)

#         # inference
#         img_pp = preproc(img)
#         out = model_inference(model_onnx_path, img_pp, device)
#         out_img, out_txt = post_process(img, out, conf_thres)
#         st.image(out_img, caption="Prediction", channels="RGB")

#         col1, col2 = st.columns(2)

#         col1.download_button(
#             label="Download prediction",
#             data=cv2.imencode(".png", out_img[..., ::-1])[1].tobytes(),
#             file_name=uploaded_file.name,
#             mime="image/png"
#         )
#         col2.download_button(
#             label="Download detections",
#             data=out_txt,
#             file_name=uploaded_file.name[:-4] + ".txt",
#             mime="text/plain"
#         )

import os
import cv2
import numpy as np
import streamlit as st
import onnxruntime as ort
from matplotlib.colors import TABLEAU_COLORS 
from pathlib import Path


ALLOWED_EXTENSIONS = {"txt", "pdf", "png", "jpg", "jpeg", "gif"}
parent_root = Path(__file__).parent.parent.absolute().__str__()
h, w = 640, 640
model_onnx_path = os.path.join(parent_root, "yolov7-p6-bonefracture.onnx")
device = "cuda"

def color_list():
    def hex2rgb(h):
        return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))
    return [hex2rgb(h) for h in TABLEAU_COLORS.values()]

colors = color_list()

def xyxy2xywhn(bbox, H, W):
    x1, y1, x2, y2 = bbox
    return [0.5*(x1+x2)/W, 0.5*(y1+y2)/H, (x2-x1)/W, (y2-y1)/H]

def xywhn2xyxy(bbox, H, W):
    x, y, w, h = bbox
    return [(x-w/2)*W, (y-h/2)*H, (x+w/2)*W, (y+h/2)*H]

def load_img(uploaded_file):
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1)
    return opencv_image[..., ::-1]

def preproc(img):
    img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
    img = img.astype(np.float32).transpose(2, 0, 1)/255
    return np.expand_dims(img, axis=0)

def model_inference(model_path, image_np, device="cpu"):
    providers = ["CUDAExecutionProvider"] if device=="cuda" else ["CPUExecutionProvider"]
    session = ort.InferenceSession(model_path, providers=providers)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    output = session.run([output_name], {input_name: image_np})
    return output[0][:, :6]

def post_process(img, output, score_threshold=0.3):
    det_bboxes, det_scores, det_labels = output[:, 0:4], output[:, 4], output[:, 5]
    id2names = {
        0: "Bone Anomaly", 1: "Bone Lesion", 2: "Foreign Body", 
        3: "Fracture", 4: "Metal", 5: "Periosteal Reaction", 
        6: "Pronator Sign", 7: "Soft Tissue", 8: "Text"
    }

    if isinstance(img, str):
        img = cv2.imread(img)
    
    img = img.astype(np.uint8)
    H, W = img.shape[:2]
    label_txt = ""
    detections = []

    for idx in range(len(det_bboxes)):
        if det_scores[idx] > score_threshold:
            bbox = det_bboxes[idx]
            label = det_labels[idx]
            
            bbox_norm = xyxy2xywhn(bbox, h, w)
            label_txt += f"{int(label)} {det_scores[idx]:.5f} {bbox_norm[0]:.5f} {bbox_norm[1]:.5f} {bbox_norm[2]:.5f} {bbox_norm[3]:.5f}\n"

            bbox_display = xywhn2xyxy(bbox_norm, H, W)
            bbox_int = [int(x) for x in bbox_display]
            x1, y1, x2, y2 = bbox_int
            color_map = colors[int(label)]
            txt = f"{id2names[label]} {det_scores[idx]:.2f}"
            
            detections.append({
                'class': id2names[label],
                'confidence': det_scores[idx],
                'bbox': bbox_int
            })
            
            (text_width, text_height), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(img, (x1, y1), (x2, y2), color_map, 2)
            cv2.rectangle(img, (x1-2, y1-text_height-10), (x1 + text_width+2, y1), color_map, -1)
            cv2.putText(img, txt, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    return img, label_txt, detections



# Page configuration
st.set_page_config(
    page_title="Bone Fracture AI",
    page_icon="🩻",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# -------------------- CUSTOM STYLING --------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background: #f5f8fc;
    }

    .block-container {
        max-width: 1400px;
        padding: 1.5rem 3rem 3rem 3rem;
    }

    /* Hide Streamlit chrome */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Hero */
    .hero {
        background: linear-gradient(135deg, #0b1f3a 0%, #123e68 55%, #176b8c 100%);
        border-radius: 24px;
        padding: 34px 40px;
        color: white;
        margin-bottom: 24px;
        box-shadow: 0 12px 35px rgba(11, 31, 58, 0.18);
        position: relative;
        overflow: hidden;
    }

    .hero:after {
        content: "✚";
        position: absolute;
        right: 45px;
        top: 15px;
        font-size: 120px;
        opacity: 0.055;
        font-weight: 800;
    }

    .hero-tag {
        display: inline-block;
        background: rgba(255,255,255,0.12);
        border: 1px solid rgba(255,255,255,0.18);
        padding: 6px 12px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
        letter-spacing: 0.5px;
        margin-bottom: 13px;
    }

    .hero h1 {
        margin: 0;
        font-size: 34px;
        font-weight: 800;
        letter-spacing: -1px;
    }

    .hero p {
        margin: 9px 0 0 0;
        font-size: 15px;
        color: rgba(255,255,255,0.78);
        max-width: 650px;
        line-height: 1.6;
    }

    /* Section heading */
    .section-title {
        color: #102a43;
        font-size: 19px;
        font-weight: 750;
        margin: 8px 0 12px 0;
    }

    .section-subtitle {
        color: #718096;
        font-size: 13px;
        margin-top: -7px;
        margin-bottom: 16px;
    }

    /* Upload card */
    .upload-card {
        background: white;
        border: 1px solid #e5ebf2;
        border-radius: 18px;
        padding: 22px 24px;
        box-shadow: 0 5px 20px rgba(31, 45, 61, 0.05);
        margin-bottom: 22px;
    }

    /* Model cards */
    .info-card {
        background: linear-gradient(145deg, #ffffff 0%, #f5f9fc 100%);
        border: 1px solid #cbdde9;
        border-left: 4px solid #2b82b8;
        border-radius: 16px;
        padding: 17px 18px;
        min-height: 92px;
        box-shadow: 0 4px 15px rgba(31, 45, 61, 0.04);
    }

    .info-label {
        color: #8291a5;
        font-size: 11px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.7px;
        margin-bottom: 6px;
    }

    .info-value {
        color: #17324d;
        font-size: 17px;
        font-weight: 750;
    }

    /* Main panels */
    .panel {
        background: white;
        border: 1px solid #d5e1eb;
        border-top: 4px solid #287bb5;
        border-radius: 20px;
        padding: 20px;
        box-shadow: 0 5px 20px rgba(31, 45, 61, 0.05);
        height: 100%;
    }

    .panel-title {
        color: #16324f;
        font-size: 17px;
        font-weight: 750;
        margin-bottom: 14px;
    }

    /* Status */
    .status {
        border-radius: 15px;
        padding: 17px 18px;
        margin-bottom: 16px;
        border: 1px solid;
    }

    .status-danger {
        background: linear-gradient(135deg, #fff7f7 0%, #fff1f1 100%);
        border: 1px solid #f2b8b8;
        border-left: 5px solid #dc5a5a;
        box-shadow: 0 5px 16px rgba(220, 90, 90, 0.08);
    }

    .status-safe {
        background: linear-gradient(135deg, #f3fcf8 0%, #eaf9f3 100%);
        border: 1px solid #a9dfc9;
        border-left: 5px solid #25a879;
        box-shadow: 0 5px 16px rgba(37, 168, 121, 0.08);
    }

    .status-neutral {
        background: linear-gradient(135deg, #f3f8ff 0%, #eaf3fc 100%);
        border: 1px solid #b9d5ed;
        border-left: 5px solid #287bb5;
        box-shadow: 0 5px 16px rgba(40, 123, 181, 0.09);
    }

    .status-icon {
        font-size: 22px;
        margin-right: 8px;
    }

    .status-title {
        font-size: 16px;
        font-weight: 800;
        color: #17324d;
    }

    .status-text {
        font-size: 12px;
        color: #718096;
        margin-top: 4px;
    }

    /* Detection item */
    .finding {
        background: linear-gradient(90deg, #f8fbff 0%, #f3f8fc 100%);
        border: 1px solid #cddfeb;
        border-left: 4px solid #2b82b8;
        border-radius: 13px;
        padding: 12px 14px;
        margin-bottom: 9px;
        box-shadow: 0 3px 10px rgba(43, 130, 184, 0.05);
        transition: all 0.2s ease;
    }

    .finding:hover {
        border-color: #a9c9df;
        transform: translateX(2px);
    }

    .finding-name {
        color: #203b55;
        font-size: 13px;
        font-weight: 700;
    }

    .finding-confidence {
        float: right;
        color: #0b7196;
        background: #e1f3fa;
        border: 1px solid #b9dfeb;
        padding: 3px 8px;
        border-radius: 20px;
        font-weight: 800;
        font-size: 12px;
        margin-top: -2px;
    }

    /* Metric */
    .metric-box {
        background: linear-gradient(145deg, #f7fbff 0%, #edf5fb 100%);
        border: 1px solid #c9ddea;
        border-top: 4px solid #287bb5;
        border-radius: 14px;
        padding: 14px;
        text-align: center;
        box-shadow: 0 4px 13px rgba(40, 123, 181, 0.06);
    }

    .metric-number {
        color: #185583;
        font-size: 25px;
        font-weight: 800;
    }

    .metric-label {
        color: #7b8a9d;
        font-size: 11px;
        font-weight: 600;
        margin-top: 2px;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #9aa7b6;
        font-size: 11px;
        margin-top: 28px;
        padding-top: 18px;
        border-top: 1px solid #e5ebf2;
    }

    /* Streamlit uploader */
    [data-testid="stFileUploader"] {
        background: #f8fafc;
        border: 2px dashed #cbd8e5;
        border-radius: 15px;
        padding: 8px;
    }

    [data-testid="stFileUploader"]:hover {
        border-color: #176b8c;
        background: #f5fbfd;
    }

    /* Buttons */
    .stDownloadButton > button {
        border-radius: 10px !important;
        font-weight: 650 !important;
    }

    /* Slider */
    [data-testid="stSlider"] {
        padding-top: 3px;
    }

    /* Image */
    [data-testid="stImage"] img {
        border-radius: 14px;
    }

    /* Expander */
    [data-testid="stExpander"] {
        border: 1px solid #bfd2e2 !important;
        border-left: 4px solid #287bb5 !important;
        border-radius: 12px !important;
        overflow: hidden;
        background: #f8fbfe !important;
        box-shadow: 0 3px 10px rgba(40, 123, 181, 0.05);
    }

    [data-testid="stExpander"] summary {
        color: #23435f !important;
        font-weight: 650 !important;
    }
</style>
""", unsafe_allow_html=True)


# -------------------- HERO --------------------
st.markdown("""
<div class="hero">
    <div class="hero-tag">🩻 AI-POWERED MEDICAL IMAGING</div>
    <h1>Bone Fracture Detection</h1>
    <p>
        Automated X-ray analysis using a YOLOv7-P6 object detection model
        to identify and localize bone abnormalities.
    </p>
</div>
""", unsafe_allow_html=True)


# -------------------- UPLOAD SECTION --------------------
st.markdown('<div class="section-title">Analyze an X-ray</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-subtitle">Upload an X-ray image and let the model identify regions of interest.</div>',
    unsafe_allow_html=True
)

upload_col, settings_col = st.columns([1.65, 0.75], gap="large")

with upload_col:
    st.markdown('<div class="upload-card">', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Drop your X-ray image here",
        type=["jpg", "jpeg", "png"],
        help="Supported formats: JPG, JPEG and PNG"
    )
    st.markdown('</div>', unsafe_allow_html=True)

with settings_col:
    st.markdown('<div class="upload-card">', unsafe_allow_html=True)
    st.markdown("**Detection Settings**")
    confidence_threshold = st.slider(
        "Confidence threshold",
        min_value=0.20,
        max_value=1.00,
        value=0.30,
        step=0.05
    )
    st.caption(f"Only predictions ≥ **{confidence_threshold:.2f}** are displayed.")
    st.markdown('</div>', unsafe_allow_html=True)


# -------------------- MODEL INFO --------------------
st.markdown('<div class="section-title">Model Information</div>', unsafe_allow_html=True)

m1, m2, m3 = st.columns(3, gap="medium")

with m1:
    st.markdown("""
    <div class="info-card">
        <div class="info-label">Detection Model</div>
        <div class="info-value">YOLOv7-P6</div>
    </div>
    """, unsafe_allow_html=True)

with m2:
    st.markdown("""
    <div class="info-card">
        <div class="info-label">Input Resolution</div>
        <div class="info-value">640 × 640</div>
    </div>
    """, unsafe_allow_html=True)

with m3:
    st.markdown("""
    <div class="info-card">
        <div class="info-label">Inference Engine</div>
        <div class="info-value">ONNX Runtime</div>
    </div>
    """, unsafe_allow_html=True)


# -------------------- ANALYSIS --------------------
if uploaded_file is not None:

    # Read the uploaded Streamlit file safely.
    # getvalue() avoids an empty buffer if the file pointer has already been read.
    file_bytes = np.frombuffer(uploaded_file.getvalue(), dtype=np.uint8)

    if file_bytes.size == 0:
        st.error("The uploaded file is empty. Please upload the X-ray image again.")
        st.stop()

    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if image is None:
        st.error("OpenCV could not decode this image. Please upload a valid JPG, JPEG, or PNG file.")
        st.stop()

    with st.spinner("Analyzing X-ray..."):
        # Use the same inference pipeline as the original application.
        img = load_img(uploaded_file)
        img_pp = preproc(img)
        out = model_inference(model_onnx_path, img_pp, device)
        annotated_image, out_txt, detections = post_process(
            img, out, confidence_threshold
        )

    # -------------------- RESULT PANELS --------------------
    st.markdown('<div style="height:18px"></div>', unsafe_allow_html=True)

    left, right = st.columns([1.35, 0.85], gap="large")

    with left:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">🩻 Analyzed X-ray</div>', unsafe_allow_html=True)

        display_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
        st.image(display_image, use_container_width=True)

        img_col, result_col = st.columns(2)

        with img_col:
            success, encoded = cv2.imencode(
                ".png", cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
            )
            if success:
                st.download_button(
                    "⬇ Download Annotated X-ray",
                    data=encoded.tobytes(),
                    file_name="bone_fracture_result.png",
                    mime="image/png",
                    use_container_width=True
                )

        with result_col:
            result_text = "Bone Fracture Detection Results\n"
            result_text += "=" * 35 + "\n\n"

            if detections:
                for det in detections:
                    result_text += (
                        f"{det.get('class', 'Unknown')}: "
                        f"{float(det.get('confidence', 0)):.2%}\n"
                    )
            else:
                result_text += "No detections found above the selected threshold.\n"

            st.download_button(
                "⬇ Download Results",
                data=result_text,
                file_name="detection_results.txt",
                mime="text/plain",
                use_container_width=True
            )

        st.markdown('</div>', unsafe_allow_html=True)

    with right:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">🔎 Detection Results</div>', unsafe_allow_html=True)

        fracture_found = any(
            str(d.get("class", "")).lower() == "fracture"
            for d in detections
        )

        if fracture_found:
            st.markdown("""
            <div class="status status-danger">
                <span class="status-icon">⚠️</span>
                <span class="status-title">Fracture Detected</span>
                <div class="status-text">
                    A region classified as fracture was detected in the X-ray.
                </div>
            </div>
            """, unsafe_allow_html=True)
        elif detections:
            st.markdown("""
            <div class="status status-neutral">
                <span class="status-icon">🔎</span>
                <span class="status-title">Abnormality Detected</span>
                <div class="status-text">
                    The model identified one or more regions of interest.
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="status status-safe">
                <span class="status-icon">✓</span>
                <span class="status-title">No Detection</span>
                <div class="status-text">
                    No regions were detected above the selected threshold.
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Metrics
        max_conf = max(
            [float(d.get("confidence", 0)) for d in detections],
            default=0
        )

        mc1, mc2 = st.columns(2)

        with mc1:
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-number">{len(detections)}</div>
                <div class="metric-label">REGIONS DETECTED</div>
            </div>
            """, unsafe_allow_html=True)

        with mc2:
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-number">{max_conf:.0%}</div>
                <div class="metric-label">TOP CONFIDENCE</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('<div style="height:18px"></div>', unsafe_allow_html=True)

        # Group findings by class
        findings = {}
        for det in detections:
            name = str(det.get("class", "Unknown"))
            conf = float(det.get("confidence", 0))
            findings[name] = max(findings.get(name, 0), conf)

        if findings:
            st.markdown("**Detected Findings**")

            for name, conf in sorted(
                findings.items(),
                key=lambda item: item[1],
                reverse=True
            ):
                st.markdown(f"""
                <div class="finding">
                    <span class="finding-name">{name}</span>
                    <span class="finding-confidence">{conf:.0%}</span>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.caption("No findings to display.")

        # Detailed bounding boxes
        if detections:
            with st.expander("View detailed detection data"):
                for i, det in enumerate(detections, 1):
                    st.markdown(
                        f"**Detection {i}:** "
                        f"{det.get('class', 'Unknown')} — "
                        f"{float(det.get('confidence', 0)):.2%}"
                    )
                    if det.get("bbox"):
                        st.caption(f"Bounding box: {det.get('bbox')}")

        st.markdown('</div>', unsafe_allow_html=True)


# -------------------- FOOTER --------------------
st.markdown("""
<div class="footer">
    Bone Fracture Detection System &nbsp;•&nbsp; YOLOv7-P6 &nbsp;•&nbsp;
    AI-assisted X-ray analysis
</div>
""", unsafe_allow_html=True)
