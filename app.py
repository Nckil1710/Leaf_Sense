# app.py - Mobile-first one-click capture + PC webcam + upload fallback
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
from PIL import Image, ImageOps, ExifTags

st.set_page_config(page_title="LeafSense: Disease Detection & Expert System", layout="wide")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- Gemini setup (same as before) ----
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"] if "GEMINI_API_KEY" in st.secrets else os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY is None:
    st.error("No Gemini API key configured. Please set GEMINI_API_KEY.")
    st.stop()
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-2.5-flash')

# ---- Cached loaders (unchanged) ----
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

# ---- Pipeline helpers (unchanged logic but kept local) ----
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
    return {"expert_system_output": expert_advice, "faiss_retrieved": supporting_context, "retrieval_quality": "High"}

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
        t_response = model.generate_content("Translate into clear spoken Telugu, only actionable, short advice, nothing extra:\n\n" + short_advice)
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
        t_response = model.generate_content("Translate into clear spoken Telugu, only actionable, short advice, nothing extra:\n\n" + short_summary)
        return t_response.text.strip() if hasattr(t_response, "text") else str(t_response)
    return short_summary

def preprocess_image(img, size=(256, 320)):
    transform = A.Compose([A.Resize(*size), A.Normalize(mean=[0.0,0.0,0.0], std=[1.0,1.0,1.0]), ToTensorV2()])
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

# ---- image utilities: server-side autorotate + compress (safe) ----
def pil_autorotate(img: Image.Image) -> Image.Image:
    try:
        return ImageOps.exif_transpose(img)
    except Exception:
        return img

def compress_image_bytes(input_bytes: bytes, max_dim: int = 1024, quality: int = 85) -> np.ndarray:
    """
    Convert raw bytes -> OpenCV BGR image:
      - auto-rotate using EXIF
      - downscale so longest side <= max_dim
      - re-encode to JPEG with given quality to reduce size
    """
    try:
        img = Image.open(BytesIO(input_bytes)).convert("RGB")
    except Exception as e:
        raise ValueError(f"Could not open image: {e}")

    img = pil_autorotate(img)
    w, h = img.size
    if max(w, h) > max_dim:
        scale = max_dim / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

    out = BytesIO()
    img.save(out, format='JPEG', quality=quality, optimize=True)
    arr = np.frombuffer(out.getvalue(), dtype=np.uint8)
    cv_img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return cv_img

# ---- client-side camera HTML (mobile-friendly file input with capture=environment).
# This input will open back camera on most mobile browsers and returns file immediately.
CAMERA_HTML_MOBILE = """
<div style="display:flex; gap:8px; align-items:center;">
  <label style="background:#1976d2;color:#fff;padding:10px 14px;border-radius:8px;cursor:pointer;font-weight:600;">
    📷 Use Camera (mobile)
    <input id="mfile" accept="image/*" capture="environment" type="file" style="display:none" />
  </label>
  <span style="color:#666;font-size:0.9em;">(opens back camera on most phones)</span>
</div>
<script>
document.getElementById('mfile').onchange = function(e){
  // nothing else — file will be returned to Streamlit via normal file_uploader path
};
</script>
"""

# ---- simple inline webcam widget for PC (capture and write data URL into textarea for paste) ----
WEBCAM_HTML_PC = r"""
<div style="font-size:14px; margin-bottom:6px;">Inline webcam (desktop): start, capture, then press 'Use Captured' below</div>
<video id="v" width="420" height="315" autoplay playsinline style="border:1px solid #ddd;"></video>
<br/>
<button onclick="startCam()" style="padding:6px 10px;border-radius:6px;background:#1976d2;color:#fff;border:none;">Start</button>
<button onclick="snap()" style="padding:6px 10px;border-radius:6px;background:#388e3c;color:#fff;border:none;margin-left:6px;">Capture</button>
<canvas id="c" width="420" height="315" style="display:none;"></canvas>
<script>
async function startCam(){
  try {
    const s = await navigator.mediaDevices.getUserMedia({video:true, audio:false});
    document.getElementById('v').srcObject = s;
  } catch(e){ alert('Camera error: '+e.message); }
}
function snap(){
  const v=document.getElementById('v'), c=document.getElementById('c');
  c.width = v.videoWidth; c.height = v.videoHeight;
  const ctx = c.getContext('2d'); ctx.drawImage(v,0,0);
  const data = c.toDataURL('image/jpeg',0.9);
  // write data URL into document.title (some browsers may keep it) and post message to parent
  document.title = data;
  window.parent.postMessage({isStreamlitImage:true, data}, "*");
  // also populate an auto textarea if exists (Streamlit will show a paste box below)
  const ta = document.getElementById('capturedDataTextarea');
  if(ta){ ta.value = data; }
  alert('Captured. If the app auto-imports, you will see the preview. Otherwise paste the data URL into the paste box below.');
}
</script>
"""

def dataurl_to_cv2_img(data_url: str) -> np.ndarray:
    header, encoded = data_url.split(',', 1)
    data = base64.b64decode(encoded)
    return compress_image_bytes(data, max_dim=1024, quality=85)  # reuse compression for safety

# ---- main app ----
def main():
    st.markdown("<h1 style='text-align:center;color:#1976d2;'>🌿 LeafSense Disease Detection & Expert System</h1>", unsafe_allow_html=True)

    # Sidebar: mode selection UI
    with st.sidebar:
        st.markdown("## 🌱 Input Mode")
        mode_choice = st.radio("Choose input mode", ["Camera → Mobile", "Camera → PC", "Upload from Device"])
        st.write("---")
        st.markdown("Tips:")
        st.write("- Mobile: tap *Use Camera (mobile)* and take the photo (one step).")
        st.write("- PC: use inline webcam capture (start → capture → Use Captured).")
        st.write("- Upload: browse device files as normal.")

    lang = st.selectbox("Choose Output Language:", ["English", "Telugu"])

    # Load models/resources
    with st.spinner("Loading models & FAISS index..."):
        yolo_model = load_yolo_model("leafsense_best.pt")
        unet_model = load_unet_model("best_weights.pth")
        densenet_model = load_densenet_model("densenet121_final_model.h5")
        class_names = load_class_names("class_names.npy")
        knowledge_base = load_knowledge_base("expert_knowledge_base.json")
        faiss_index = load_faiss_index("faiss_index.bin")
        faiss_metadata = load_faiss_metadata("faiss_metadata.pkl")
        encoder = load_encoder()

    img = None
    source_type = None

    # -- Mobile camera flow (recommended one-step) --
    if mode_choice == "Camera → Mobile":
        st.markdown("### Mobile camera (one-step). Tap the button to open the phone camera.")
        # show embedded HTML file input requesting back camera
        components.html(CAMERA_HTML_MOBILE, height=80)
        # Also show a file_uploader so phones that expose camera via picker will return file immediately
        uploaded_file = st.file_uploader("Or use this to capture/upload (recommended for mobile)", type=['jpg','jpeg','png'])
        if uploaded_file is not None:
            raw = uploaded_file.read()
            try:
                img = compress_image_bytes(raw, max_dim=1024, quality=85)
                source_type = "Mobile camera (file_uploader)"
            except Exception as e:
                st.error(f"Image decoding error: {e}")
                st.stop()

    # -- PC camera flow (desktop) --
    elif mode_choice == "Camera → PC":
        st.markdown("### PC webcam: start the camera (allow permission), capture, then click 'Use Captured' below to import.")
        components.html(WEBCAM_HTML_PC, height=380)
        # show a paste box that JS tries to fill automatically; also allow manual paste
        pasted = st.text_area("Captured data URL (auto-filled if allowed) — otherwise paste it here", key="capturedData", height=120)
        if pasted and pasted.strip().startswith("data:image"):
            try:
                img = dataurl_to_cv2_img(pasted.strip())
                source_type = "PC webcam (captured)"
            except Exception as e:
                st.error("Failed to decode pasted data URL.")
                st.stop()

        # also allow normal file upload fallback
        uploaded_file_pc = st.file_uploader("Or upload an image file (fallback)", type=['jpg','jpeg','png'], key="pc_upload")
        if (img is None) and (uploaded_file_pc is not None):
            raw = uploaded_file_pc.read()
            try:
                img = compress_image_bytes(raw, max_dim=1024, quality=85)
                source_type = "PC upload fallback"
            except Exception as e:
                st.error(f"Decoding error: {e}")
                st.stop()

    # -- Upload from device mode --
    else:
        st.markdown("### Upload from device")
        uploaded = st.file_uploader("Upload image", type=['jpg','jpeg','png'])
        if uploaded is not None:
            raw = uploaded.read()
            try:
                img = compress_image_bytes(raw, max_dim=1024, quality=85)
                source_type = "Uploaded file"
            except Exception as e:
                st.error(f"Image decode error: {e}")
                st.stop()

    # If no image, stop and wait
    if img is None:
        st.info("No image yet — capture or upload a photo to run the pipeline.")
        st.stop()

    # show preview immediately
    st.markdown("#### Input image preview")
    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), width=360, caption=source_type)

    # ===== PHASE 1: Leaf segmentation (YOLO) =====
    with st.spinner("PHASE 1: Segmenting leaf (YOLO)..."):
        try:
            yres = yolo_model(img)
        except Exception as e:
            st.error(f"YOLO model error: {e}")
            st.stop()
        if (not hasattr(yres[0], "masks")) or (yres[0].masks is None):
            st.error("No leaf detected. Try again with a clearer leaf photo.")
            st.stop()
        leaf_mask = yres[0].masks.data.cpu().numpy()[0]
    st.success("✅ Phase 1 Complete")

    # ===== PHASE 2: Unet disease mask =====
    with st.spinner("PHASE 2: Segmenting disease (UNet)..."):
        img_tensor = preprocess_image(img).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            out = unet_model(img_tensor)
            pred_prob = torch.sigmoid(out)
            pred_mask = (pred_prob > 0.5).float().cpu().squeeze().numpy()
    st.success("✅ Phase 2 Complete")

    # ===== PHASE 3: Severity + classification =====
    severity, severity_label, dpx, tpx = calculate_severity(pred_mask, leaf_mask)
    st.markdown(f"<h2 style='color:#d84315;'>Severity: {severity:.2f}% ({severity_label})</h2>", unsafe_allow_html=True)
    disease_class, pred_conf = "Unknown", 0.0
    if densenet_model is not None and class_names is not None:
        with st.spinner("PHASE 3: Classifying disease..."):
            IMG_SIZE = 224
            img_tf = tf.image.resize_with_pad(tf.image.decode_image(cv2.imencode('.jpg', img)[1].tobytes(), channels=3), IMG_SIZE, IMG_SIZE)
            img_tf = tf.cast(img_tf, tf.float32)
            from tensorflow.keras.applications.densenet import preprocess_input
            img_tf = preprocess_input(img_tf)
            img_tf = tf.expand_dims(img_tf, 0)
            pred = densenet_model.predict(img_tf, verbose=0)
            pred_class = np.argmax(pred, axis=1)[0]
            disease_class = class_names[pred_class]
            pred_conf = float(np.max(pred))
        st.markdown(f"<h3 style='color:#388e3c;'>Disease: {disease_class.replace('___',' - ')}</h3>", unsafe_allow_html=True)
        st.markdown(f"<b>Model confidence:</b> {pred_conf:.4f}")
        st.success("✅ Phase 3 Complete")
    else:
        st.warning("DenseNet model / class names not loaded — classification skipped.")

    # ===== PHASE 4: Expert system inference =====
    with st.spinner("PHASE 4: Expert inference..."):
        expert_advice = phase4_expert_system_inference(disease_class, severity_label, knowledge_base)
    if "error" in expert_advice:
        st.warning(expert_advice["error"])
        st.stop()
    st.success("✅ Phase 4 Complete")

    # ===== PHASE 5: RAG + Gemini =====
    with st.spinner("PHASE 5: RAG + LLM summary..."):
        rag_ctx = phase5_rag_with_faiss(expert_advice, faiss_index, faiss_metadata, encoder)
        gemini_summary = generate_gemini_recommendation_phase5(expert_advice, rag_ctx, lang)
    st.success("✅ Phase 5 Complete")

    # Display results & visualizations
    st.header("🌱 Short AI Summary")
    st.markdown(gemini_summary)

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
    axes[3].set_title(f"Overlay & Prediction\n{disease_class.replace('___',' - ')}\n{severity:.2f}% ({severity_label})")
    axes[3].axis('off')
    st.pyplot(fig)

    # JSON report download
    results = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "phase_1_segmentation": "Leaf segmented",
        "phase_2_disease_segmentation": "Disease area identified",
        "phase_3_classification": {"disease": disease_class, "confidence": pred_conf, "severity": severity_label},
        "phase_4_expert_inference": expert_advice,
        "phase_5_rag_output": {"gemini_summary": gemini_summary, "language": lang},
    }
    st.download_button("⬇️ Download Full Report (JSON)", json.dumps(results, indent=2), file_name="leafsense_analysis_report.json", mime="application/json")

if __name__ == "__main__":
    main()
