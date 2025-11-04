# app.py
import streamlit as st
import streamlit.components.v1 as components
import torch
import segmentation_models_pytorch as smp
from ultralytics import YOLO
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize
import tensorflow as tf
from tensorflow.keras.models import load_model
from keras.utils import custom_object_scope
import json
import os
import pickle
import faiss
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import warnings
warnings.filterwarnings('ignore')
from datetime import datetime
import base64
from io import BytesIO
from PIL import Image

st.set_page_config(page_title="LeafSense: Disease Detection & Expert System", layout="wide")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==== Gemini Key Setup ====
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"] if "GEMINI_API_KEY" in st.secrets else os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY is None:
    st.error("No Gemini API key configured. Please set GEMINI_API_KEY.")
    st.stop()
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-2.5-flash')

# --- Cache Loaders ---
@st.cache_resource(show_spinner=True)
def load_yolo_model(path): return YOLO(path)

@st.cache_resource(show_spinner=True)
def load_unet_model(path):
    m = smp.from_pretrained(path)
    m.to(DEVICE).eval()
    return m

@st.cache_resource(show_spinner=True)
def load_densenet_model(path):
    class DummyCast(tf.keras.layers.Layer):
        def __init__(self, dtype=None, **kwargs):
            super().__init__(**kwargs)
            self._dtype = dtype
        def call(self, inputs): return inputs
        def get_config(self):
            config = super().get_config()
            config.update({"dtype": self._dtype})
            return config
    try:
        with custom_object_scope({'Cast': DummyCast, 'UniqueCast': DummyCast}):
            return load_model(path, custom_objects={'Cast': DummyCast, 'UniqueCast': DummyCast}, compile=False)
    except Exception as e:
        st.error(f"Model load error: {e}")
        return None

@st.cache_data(show_spinner=True)
def load_class_names(path): return np.load(path)

@st.cache_data(show_spinner=True)
def load_knowledge_base(path):
    try:
        with open(path, 'r') as f: return json.load(f)
    except Exception: return {}

@st.cache_resource(show_spinner=True)
def load_faiss_index(index_path): return faiss.read_index(index_path)

@st.cache_data(show_spinner=True)
def load_faiss_metadata(metadata_path):
    with open(metadata_path, 'rb') as f: return pickle.load(f)

@st.cache_resource(show_spinner=True)
def load_encoder(): return SentenceTransformer('all-MiniLM-L6-v2')

# ---------- Utility & pipeline functions (your original ones) ----------
def normalize_class_name(name):
    return (name.replace(" ", "_").replace("-", "_").replace("__", "___").strip().lower())

def phase4_expert_system_inference(disease_class, severity_label, knowledge_base):
    normalized_kb = {normalize_class_name(k): k for k in knowledge_base}
    lookup_key = normalize_class_name(disease_class)
    if lookup_key not in normalized_kb:
        similar = [k for k in knowledge_base if disease_class.replace(" ", "_") in k]
        return {"error": f"Disease '{disease_class}' not found. Did you mean: {similar[:3]}?"}
    
    actual_key = normalized_kb[lookup_key]
    disease_info = knowledge_base[actual_key]
    severity_label = severity_label.lower().strip()
    
    if severity_label not in disease_info["severity_identification"]:
        return {"error": f"Severity '{severity_label}' not recognized.", "valid_severities": list(disease_info["severity_identification"].keys())}
    
    return {
        "disease_name": actual_key.replace("___", " - ").replace("_", " "),
        "severity_level": severity_label.upper(),
        "symptoms": disease_info["symptoms"],
        "severity_description": disease_info["severity_identification"][severity_label],
        "pesticides": disease_info["pesticides"],
        "treatment": disease_info["solutions"][severity_label],
        "prevention": disease_info["prevention"]
    }

def phase5_rag_with_faiss(expert_advice, faiss_index, faiss_metadata, encoder, top_k=2):
    query_text = f"{expert_advice['disease_name']} {expert_advice['symptoms']} {expert_advice['severity_level']}"
    query_vector = encoder.encode(query_text, convert_to_numpy=True).astype('float32').reshape(1, -1)
    distances, indices = faiss_index.search(query_vector, top_k)
    supporting_context = []
    for idx in indices[0]:
        supporting_context.append(faiss_metadata[idx])
    return {
        "expert_system_output": expert_advice,
        "faiss_retrieved": supporting_context,
        "retrieval_quality": "High"
    }

def disease_expert_advisor(disease_query, faiss_index, faiss_metadata, encoder, knowledge_base, lang="English"):
    query_vector = encoder.encode(disease_query, convert_to_numpy=True).astype('float32').reshape(1, -1)
    distances, indices = faiss_index.search(query_vector, k=3)
    results = []
    for idx in indices[0]:
        metadata = faiss_metadata[idx]
        results.append({
            "disease": metadata['disease_readable'],
            "symptoms": metadata['symptoms'],
            "pesticides": metadata['pesticides'],
            "prevention": metadata['prevention'],
            "severity_levels": metadata['severity_levels']
        })
    prompt = (
        f"A farmer asked: \"{disease_query}\"\n"
        "Give only actionable info: Disease, Severity (if possible), Immediate action, Recommended pesticides, Prevention tips, When to act/treat, nothing more.\n"
        "Here are top matching database entries:\n"
    )
    for i, result in enumerate(results, 1):
        prompt += (
            f"{i}. Disease: {result['disease']}\n"
            f"   Symptoms: {result['symptoms']}\n"
            f"   Pesticides: {result['pesticides']}\n"
            f"   Prevention: {result['prevention']}\n"
        )
    response = model.generate_content(prompt)
    short_advice = response.text.strip() if hasattr(response, "text") else str(response)
    if lang == "Telugu":
        telugu_prompt = (
            "Translate into clear spoken Telugu, only actionable, short advice, nothing extra or technical:\n\n"
            + short_advice
        )
        t_response = model.generate_content(telugu_prompt)
        return t_response.text.strip() if hasattr(t_response, "text") else str(t_response), results
    return short_advice, results

def generate_gemini_recommendation_phase5(expert_advice, rag_context, lang="English"):
    prompt = (
        f"ONLY use the expert-verified info below. Be extremely practical, keep it bullet style, show only: Disease, Severity, Immediate action, Recommended pesticides, Prevention tips, When to act/treat. Nothing else.\n"
        f"Disease: {expert_advice['disease_name']}\n"
        f"Severity: {expert_advice['severity_level']}\n"
        f"Symptoms: {expert_advice['symptoms']}\n"
        f"Severity Description: {expert_advice['severity_description']}\n"
        f"Pesticides: {expert_advice['pesticides']}\n"
        f"Treatment: {expert_advice['treatment']}\n"
        f"Prevention: {expert_advice['prevention']}\n"
    )
    response = model.generate_content(prompt)
    short_summary = response.text.strip() if hasattr(response, "text") else str(response)
    if lang == "Telugu":
        telugu_prompt = (
            "Translate into clear spoken Telugu, only actionable, short advice, nothing extra:\n\n"
            + short_summary
        )
        t_response = model.generate_content(telugu_prompt)
        return t_response.text.strip() if hasattr(t_response, "text") else str(t_response)
    return short_summary

def preprocess_image(img, size=(256, 320)):
    transform = A.Compose([
        A.Resize(*size),
        A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0]),
        ToTensorV2()
    ])
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    augmented = transform(image=img_rgb)
    return augmented['image']

def calculate_severity(disease_mask, leaf_mask):
    if leaf_mask.shape != disease_mask.shape:
        leaf_mask = resize(leaf_mask, disease_mask.shape, order=0, preserve_range=True, anti_aliasing=False)
        leaf_mask = (leaf_mask > 0.5).astype(np.uint8)
    diseased_pixels = np.sum((disease_mask > 0) & (leaf_mask > 0))
    total_pixels = np.sum(leaf_mask > 0)
    severity = (diseased_pixels / total_pixels * 100) if total_pixels > 0 else 0
    if severity < 10: label = "Mild"
    elif severity < 30: label = "Moderate"
    else: label = "Severe"
    return severity, label, diseased_pixels, total_pixels

# ------------- Camera widget HTML (switchable front/back) -------------
CAMERA_WIDGET = r"""
<style>
  .btn { padding:8px 12px; border-radius:6px; background:#1976d2; color:white; border:none; font-weight:600; cursor:pointer; margin-right:6px; }
  .controls { margin:8px 0; }
</style>
<div>
  <div class="controls">
    <select id="videoSelect" style="padding:6px;border-radius:6px;"></select>
    <button id="startBtn" class="btn">Start Camera</button>
    <button id="switchBtn" class="btn">Switch Camera</button>
    <button id="snapBtn" class="btn">Capture</button>
  </div>
  <video id="video" width="360" height="270" autoplay playsinline style="border:1px solid #ddd; background:#000;"></video>
  <canvas id="canvas" width="360" height="270" style="display:none;"></canvas>
</div>

<script>
const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const videoSelect = document.getElementById('videoSelect');
let currentStream = null;
let devices = [];

async function gotDevices(deviceInfos) {
  devices = deviceInfos.filter(d => d.kind === 'videoinput');
  videoSelect.innerHTML = '';
  devices.forEach((d,i) => {
    const option = document.createElement('option');
    option.value = d.deviceId;
    option.text = d.label || ('Camera ' + (i+1));
    videoSelect.appendChild(option);
  });
}

navigator.mediaDevices.enumerateDevices().then(gotDevices).catch(e => console.error(e));

async function startStream(deviceId) {
  if (currentStream) {
    currentStream.getTracks().forEach(track => track.stop());
  }
  const constraints = { video: deviceId ? { deviceId: { exact: deviceId } } : { facingMode: "environment" }, audio: false };
  try {
    const stream = await navigator.mediaDevices.getUserMedia(constraints);
    currentStream = stream;
    video.srcObject = stream;
  } catch (e) {
    console.error("getUserMedia() error:", e);
    alert("Camera permission or device error: " + e.message);
  }
}

document.getElementById('startBtn').onclick = async () => {
  await navigator.mediaDevices.enumerateDevices().then(gotDevices);
  const env = devices.find(d => (d.label && (d.label.toLowerCase().includes('back') || d.label.toLowerCase().includes('rear'))));
  if (env) {
    videoSelect.value = env.deviceId;
    startStream(env.deviceId);
  } else {
    startStream();
  }
};

document.getElementById('switchBtn').onclick = async () => {
  await navigator.mediaDevices.enumerateDevices().then(gotDevices);
  if (devices.length <= 1) return alert('No alternate camera found');
  const idx = devices.findIndex(d => d.deviceId === (videoSelect.value || devices[0].deviceId));
  const next = devices[(idx + 1) % devices.length];
  videoSelect.value = next.deviceId;
  startStream(next.deviceId);
};

document.getElementById('snapBtn').onclick = () => {
  const ctx = canvas.getContext('2d');
  canvas.width = video.videoWidth || 360;
  canvas.height = video.videoHeight || 270;
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
  const dataUrl = canvas.toDataURL('image/jpeg', 0.9);
  // write captured data URL into document.title (so Streamlit can read it later)
  document.title = dataUrl;
  // postMessage for robustness
  window.parent.postMessage({isStreamlitImage: true, data: dataUrl}, "*");
  alert('Captured! Now close the camera dialog and press "Get Captured Image" in the Streamlit app.');
};
</script>
"""

# ------------- Helper: convert data URL to cv2 image -------------
def dataurl_to_cv2_img(data_url):
    header, encoded = data_url.split(',', 1)
    data = base64.b64decode(encoded)
    image = Image.open(BytesIO(data)).convert("RGB")
    open_cv_image = np.array(image)[:, :, ::-1]  # RGB->BGR
    return open_cv_image

# ------------- Main app -------------
def main():
    st.markdown("<h1 style='text-align: center; color: #1976d2;'>🌿 LeafSense Disease Detection & Expert System</h1>", unsafe_allow_html=True)

    # Sidebar and mode selection
    with st.sidebar:
        st.markdown("## 🌱 Welcome to LeafSense!")
        st.markdown("### Choose Your Mode:")
        mode = st.radio("What would you like to do?", ["Image Detection", "Ask Disease Expert"])
        if mode == "Image Detection":
            st.markdown("#### 📸 Image Detection Mode")
            st.write(
                "- Upload a **clear, full photo** of a single leaf on a plain background.\n"
                "- Leaf should be visible edge-to-edge, with minimal shadow or overlap.\n"
                "- Crop/zoom so the leaf fills most of the frame for best results.\n"
            )
        else:
            st.markdown("#### 💭 Disease Expert Mode")
            st.write("- Describe the disease or symptoms you're concerned about.\n- Our AI will search our database and provide expert recommendations.")

    lang = st.selectbox("Choose Output Language:", ["English", "Telugu"])

    # Load models and resources
    with st.spinner("Loading models & FAISS index..."):
        yolo_model = load_yolo_model("leafsense_best.pt")
        unet_model = load_unet_model("best_weights.pth")
        densenet_model = load_densenet_model("densenet121_final_model.h5")
        class_names = load_class_names("class_names.npy")
        knowledge_base = load_knowledge_base("expert_knowledge_base.json")
        faiss_index = load_faiss_index("faiss_index.bin")
        faiss_metadata = load_faiss_metadata("faiss_metadata.pkl")
        encoder = load_encoder()

    if mode == "Image Detection":
        st.header("📷 Capture or Upload Leaf Image")
        st.markdown("Use the camera widget below to choose front/back camera and capture a photo. After capture, press **Get Captured Image** to import the photo into Streamlit. If device fails, use the Upload fallback.")

        # Render the camera widget in a modal-like area
        components.html(CAMERA_WIDGET, height=420)

        if st.button("Get Captured Image"):
            # Try to fetch document.title by rendering a small HTML that writes document.title into the body.
            # The HTML below returns the current document.title inside the iframe content which Streamlit shows;
            # however behavior may depend on the browser. This attempt often works to capture the data URL set earlier.
            getter_html = """
            <script>
              // write the current document.title to the html body so Streamlit can read it
              const t = document.title || '';
              document.open();
              document.write('<div id="capt">' + t + '</div>');
              document.close();
            </script>
            """
            res = components.html(getter_html, height=120)
            # The trick: try to read from the page's title via another small HTML that places it into the DOM then user copies it:
            st.info("If the image does not auto-load, please close the camera dialog and use the Upload fallback below, or paste the data URL if available.")
            # Additionally offer file uploader fallback
        uploaded_file = st.file_uploader("Upload Leaf Image (fallback)", type=['jpg', 'jpeg', 'png'])
        captured_data_url = None

        # Try reading the page title via JS posted message stored in st.session_state by some browsers
        # We attempt to read a known place: Streamlit doesn't provide a direct pipe; rely on uploaded file or manual paste.
        if uploaded_file is not None:
            input_image_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            img = cv2.imdecode(input_image_bytes, cv2.IMREAD_COLOR)
            source_type = "Uploaded file"
        else:
            # Offer an alternate text area to paste data URL if automatic capture didn't work
            paste_data = st.text_area("If you used Capture and the image didn't appear, paste the image Data URL here (starts with 'data:image/jpeg;base64,')", height=80)
            if paste_data and paste_data.strip().startswith("data:image"):
                try:
                    img = dataurl_to_cv2_img(paste_data.strip())
                    source_type = "Captured (pasted data URL)"
                except Exception as e:
                    st.error("Failed to decode pasted data URL.")
                    st.stop()
            else:
                img = None
                source_type = None

        if img is None:
            st.info("Please capture or upload a leaf image to begin.")
            st.stop()

        # Show preview
        with st.container():
            st.markdown("#### Input Image Preview")
            st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption=source_type, width=320)

        # ===== PHASE 1: Leaf Segmentation (YOLO) =====
        with st.spinner("PHASE 1: Segmenting leaf (YOLO)..."):
            yolo_results = yolo_model(img)
            if (not hasattr(yolo_results[0], "masks")) or (yolo_results[0].masks is None):
                st.error("No leaf detected. Try again with a clearer leaf photo!")
                return
            leaf_mask = yolo_results[0].masks.data.cpu().numpy()[0]
        st.success("✅ Phase 1 Complete: Leaf Segmentation")

        # ===== PHASE 2: Disease Segmentation (UNet) =====
        with st.spinner("PHASE 2: Segmenting disease (UNet)..."):
            img_tensor = preprocess_image(img).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                output = unet_model(img_tensor)
                pred_prob = torch.sigmoid(output)
                pred_mask = (pred_prob > 0.5).float().cpu().squeeze().numpy()
        st.success("✅ Phase 2 Complete: Disease Segmentation")

        # ===== PHASE 3: Severity Calculation + Disease Classification =====
        severity, severity_label, diseased_px, total_px = calculate_severity(pred_mask, leaf_mask)
        st.markdown(f"<h2 style='color:#d84315;'>Severity: {severity:.2f}% ({severity_label})</h2>", unsafe_allow_html=True)
        disease_class, pred_confidence = "Unknown", 0.0
        if densenet_model is not None and class_names is not None:
            with st.spinner("PHASE 3: Classifying disease (DenseNet121)..."):
                IMG_SIZE = 224
                # encode image to jpeg -> use tensorflow decode (existing code)
                img_tf = tf.image.resize_with_pad(tf.image.decode_image(cv2.imencode('.jpg', img)[1].tobytes(), channels=3), IMG_SIZE, IMG_SIZE)
                img_tf = tf.cast(img_tf, tf.float32)
                from tensorflow.keras.applications.densenet import preprocess_input
                img_tf = preprocess_input(img_tf)
                img_tf = tf.expand_dims(img_tf, 0)
                pred = densenet_model.predict(img_tf, verbose=0)
                pred_class = np.argmax(pred, axis=1)[0]
                disease_class = class_names[pred_class]
                pred_confidence = float(np.max(pred))
            st.markdown(f"<h3 style='color:#388e3c;'>Disease: {disease_class.replace('___', ' - ')}</h3>", unsafe_allow_html=True)
            st.markdown(f"<b>Model confidence:</b> {pred_confidence:.4f}")
            st.success("✅ Phase 3 Complete: Disease Classification")
        else:
            st.warning("DenseNet model or class names not loaded. Classification skipped.")

        # ===== PHASE 4: EXPERT SYSTEM INFERENCE ENGINE =====
        with st.spinner("PHASE 4: Expert System Inference (KB Verification)..."):
            expert_advice = phase4_expert_system_inference(disease_class, severity_label, knowledge_base)
        if "error" in expert_advice:
            st.warning(expert_advice["error"])
            return
        st.success("✅ Phase 4 Complete: Expert System Inference")

        # ===== PHASE 5: RAG + FAISS + GEMINI =====
        with st.spinner("PHASE 5: RAG Retrieval (FAISS + Gemini LLM)..."):
            rag_context = phase5_rag_with_faiss(expert_advice, faiss_index, faiss_metadata, encoder)
            gemini_summary = generate_gemini_recommendation_phase5(expert_advice, rag_context, lang)
        st.success("✅ Phase 5 Complete: RAG + LLM Report")

        # --- DISPLAY SHORT AI SUMMARY ---
        st.header("🌱 Short AI Summary")
        st.markdown(gemini_summary)

        # Detailed expander and visualizations (unchanged)
        with st.expander("See Full Detailed System Report"):
            st.markdown(f"**Disease:** {expert_advice['disease_name']}")
            st.markdown(f"**Severity Level:** {expert_advice['severity_level']}")
            st.markdown(f"**Symptoms:** {expert_advice['symptoms']}")
            st.markdown(f"**Severity Description:** {expert_advice['severity_description']}")
            st.markdown("**Pesticides Recommended:**")
            for p in expert_advice['pesticides'].split(","):
                st.markdown(f"- {p.strip()}")
            st.markdown(f"**Treatment:** {expert_advice['treatment']}")
            st.markdown(f"**Prevention:**")
            for idx, line in enumerate(expert_advice['prevention'].split(". ")):
                if line.strip():
                    st.markdown(f"{idx+1}. {line.strip()}.")

        # Visualization
        st.header("📊 Disease Prediction and Segmentation")
        fig, axes = plt.subplots(1, 4, figsize=(18, 6))
        image_vis = cv2.cvtColor(cv2.resize(img, (pred_mask.shape[1], pred_mask.shape[0])), cv2.COLOR_BGR2RGB)
        axes[0].imshow(image_vis); axes[0].set_title("Original Image"); axes[0].axis('off')
        axes[1].imshow(leaf_mask, cmap='Greens', alpha=0.8); axes[1].set_title("Leaf Mask (YOLO)"); axes[1].axis('off')
        axes[2].imshow(pred_mask, cmap='autumn', alpha=0.8); axes[2].set_title("Disease Mask (UNet)"); axes[2].axis('off')
        overlay = np.zeros_like(pred_mask)
        overlay[(pred_mask > 0) & (resize(leaf_mask, pred_mask.shape, order=0, preserve_range=True, anti_aliasing=False) > 0)] = 1
        axes[3].imshow(image_vis)
        axes[3].imshow(overlay, cmap='autumn', alpha=0.5)
        axes[3].set_title(f"Overlay & Prediction\n{disease_class.replace('___', ' - ')}\n{severity:.2f}% ({severity_label})")
        axes[3].axis('off')
        st.pyplot(fig)

        # Download report
        results = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "phase_1_segmentation": "Leaf segmented",
            "phase_2_disease_segmentation": "Disease area identified",
            "phase_3_classification": {"disease": disease_class, "confidence": pred_confidence, "severity": severity_label},
            "phase_4_expert_inference": expert_advice,
            "phase_5_rag_output": {"gemini_summary": gemini_summary, "language": lang},
        }
        json_out = json.dumps(results, indent=2)
        st.download_button("⬇️ Download Full Report (JSON)", json_out, file_name="leafsense_analysis_report.json", mime="application/json")

    else:
        # Expert advisor mode (unchanged)
        st.header("💭 Disease Expert Advisor")
        st.markdown("---")
        st.markdown("### Ask About Any Plant Disease")
        disease_query = st.text_area("Describe your disease/symptom concern:", placeholder="e.g., 'My tomato leaves have brown spots with yellow rings' or 'What is early blight?'", height=100)
        if st.button("🔍 Get Expert Recommendation", key="expert_btn"):
            if not disease_query.strip():
                st.warning("Please describe a disease or symptom!")
            else:
                with st.spinner("Searching database and generating recommendations..."):
                    recommendation, matched_diseases = disease_expert_advisor(
                        disease_query,
                        faiss_index,
                        faiss_metadata,
                        encoder,
                        knowledge_base,
                        lang
                    )
                st.success("✅ Expert Recommendation Generated!")
                st.header("🌱 Short AI Summary")
                st.markdown(recommendation)
                st.markdown("---")
                st.header("📚 Top Matching Diseases in Database")
                for i, disease in enumerate(matched_diseases, 1):
                    with st.expander(f"{i}. {disease['disease']}"):
                        st.markdown(f"**Symptoms:** {disease['symptoms']}")
                        st.markdown(f"**Pesticides:** {disease['pesticides']}")
                        st.markdown(f"**Prevention:** {disease['prevention']}")
                        st.markdown(f"**Severity Levels:** {', '.join(disease['severity_levels'])}")

if __name__ == "__main__":
    main()
