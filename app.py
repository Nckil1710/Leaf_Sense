# app.py - Optimized Mobile-First Image Input with Gallery & Camera Support
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

# ==== Gemini setup ====
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"] if "GEMINI_API_KEY" in st.secrets else os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY is None:
    st.error("No Gemini API key configured. Please set GEMINI_API_KEY.")
    st.stop()
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-2.5-flash')

# ==== Cached loaders ====
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

# ==== Pipeline helpers ====
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
        context = faiss_metadata[idx].copy()
        supporting_context.append(context)
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

# ==== Image processing utilities ====
def pil_autorotate(img: Image.Image) -> Image.Image:
    """Auto-rotate image based on EXIF data"""
    try:
        return ImageOps.exif_transpose(img)
    except Exception:
        return img

def compress_image_bytes(input_bytes: bytes, max_dim: int = 1024, quality: int = 85) -> np.ndarray:
    """
    Convert raw bytes -> OpenCV BGR image:
      - auto-rotate using EXIF
      - downscale if needed
      - re-encode to reduce file size
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

def dataurl_to_cv2_img(data_url: str) -> np.ndarray:
    """Convert data URL to OpenCV image"""
    try:
        header, encoded = data_url.split(',', 1)
        data = base64.b64decode(encoded)
        return compress_image_bytes(data, max_dim=1024, quality=85)
    except Exception as e:
        raise ValueError(f"Failed to decode data URL: {e}")

# ==== Mobile-friendly HTML components ====

# Mobile: Direct camera + gallery picker (one-step)
MOBILE_INPUT_HTML = """
<div style="display:flex; flex-direction:column; gap:10px; align-items:center; padding:10px;">
  <div style="background:#f5f5f5; padding:12px; border-radius:8px; text-align:center;">
    <p style="margin:0; color:#333; font-weight:600;">📱 Quick Image Input</p>
  </div>
  
  <label style="background:#1976d2;color:#fff;padding:12px 16px;border-radius:8px;cursor:pointer;font-weight:600; width:100%; text-align:center;">
    📷 Open Camera / Gallery
    <input id="mobile_input" accept="image/*" capture type="file" style="display:none" />
  </label>
  
  <p style="color:#666;font-size:0.85em; margin:0; text-align:center;">
    (Mobile: tap for camera/gallery • Desktop: browse files)
  </p>
</div>
"""

# PC: Inline webcam widget
PC_WEBCAM_HTML = r"""
<div style="background:#f5f5f5; padding:15px; border-radius:8px;">
  <p style="margin:0 0 10px 0; color:#333; font-weight:600;">🖥️ Desktop Webcam Capture</p>
  <p style="margin:0 0 10px 0; font-size:0.9em; color:#666;">Start → Allow camera → Capture → Use Captured</p>
  
  <div style="border:2px solid #1976d2; border-radius:6px; overflow:hidden; margin-bottom:10px;">
    <video id="webcam" width="100%" height="300" autoplay playsinline style="background:#000; display:block;"></video>
  </div>
  
  <canvas id="canvas" width="640" height="480" style="display:none;"></canvas>
  
  <div style="display:flex; gap:8px; margin-bottom:10px;">
    <button onclick="startWebcam()" style="flex:1; padding:8px; background:#1976d2; color:#fff; border:none; border-radius:6px; font-weight:600; cursor:pointer;">▶ Start Camera</button>
    <button onclick="captureWebcam()" style="flex:1; padding:8px; background:#388e3c; color:#fff; border:none; border-radius:6px; font-weight:600; cursor:pointer;">📸 Capture</button>
  </div>
  
  <script>
  async function startWebcam(){
    try {
      const stream = await navigator.mediaDevices.getUserMedia({video: {facingMode:'user'}, audio: false});
      document.getElementById('webcam').srcObject = stream;
    } catch(e){ 
      alert('Camera access denied: '+e.message); 
    }
  }
  
  function captureWebcam(){
    const v = document.getElementById('webcam');
    const c = document.getElementById('canvas');
    c.width = v.videoWidth;
    c.height = v.videoHeight;
    const ctx = c.getContext('2d');
    ctx.drawImage(v, 0, 0);
    
    const dataUrl = c.toDataURL('image/jpeg', 0.9);
    // Write to hidden textarea for Streamlit to read
    const ta = document.getElementById('webcam_data_textarea');
    if(ta){ 
      ta.value = dataUrl; 
      ta.dispatchEvent(new Event('change', {bubbles: true}));
    }
    alert('✅ Captured! Data has been sent to the app.');
  }
  </script>
</div>
"""

# ==== Main app ====
def main():
    st.markdown("<h1 style='text-align:center;color:#1976d2;'>🌿 LeafSense Disease Detection & Expert System</h1>", unsafe_allow_html=True)

    # Sidebar: mode selection
    with st.sidebar:
        st.markdown("## 🌱 Input Mode Selection")
        input_mode = st.radio(
            "Choose how to input your leaf image:",
            ["📱 Mobile (Camera/Gallery)", "🖥️ PC Webcam", "📤 Upload File"],
            help="Mobile: one-step camera/gallery • PC: webcam or file upload • Upload: browse your device"
        )
        st.markdown("---")
        st.markdown("### 💡 Tips:")
        st.write("✓ Clear, close-up leaf photos work best")
        st.write("✓ Plain background recommended")
        st.write("✓ Single leaf per image")

    lang = st.selectbox("📝 Output Language:", ["English", "Telugu"])

    # Load all models upfront
    with st.spinner("Loading models & AI resources..."):
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

    # ==== INPUT: Mobile (camera + gallery in one) ====
    if input_mode == "📱 Mobile (Camera/Gallery)":
        st.markdown("### One-Step Image Capture or Upload")
        components.html(MOBILE_INPUT_HTML, height=130)
        
        # Standard file uploader (works for both mobile and desktop as fallback)
        uploaded_file = st.file_uploader("Or select from files:", type=['jpg', 'jpeg', 'png'], key="mobile_upload")
        if uploaded_file is not None:
            try:
                img = compress_image_bytes(uploaded_file.read(), max_dim=1024, quality=85)
                source_type = "📱 Mobile Input"
            except Exception as e:
                st.error(f"Image processing error: {e}")
                st.stop()

    # ==== INPUT: PC Webcam ====
    elif input_mode == "🖥️ PC Webcam":
        st.markdown("### Desktop Webcam Capture")
        components.html(PC_WEBCAM_HTML, height=420)
        
        # Hidden textarea to capture webcam data URL
        webcam_data = st.text_area("Webcam capture data (auto-filled):", key="webcam_data_textarea", height=1, disabled=True)
        
        if webcam_data and webcam_data.startswith("data:image"):
            try:
                img = dataurl_to_cv2_img(webcam_data)
                source_type = "🖥️ PC Webcam"
            except Exception as e:
                st.error(f"Failed to decode webcam capture: {e}")
                st.stop()
        
        # Fallback: file upload
        st.markdown("**Fallback:** No webcam data? Upload an image file:")
        uploaded_file = st.file_uploader("Upload image:", type=['jpg', 'jpeg', 'png'], key="pc_upload")
        if (img is None) and (uploaded_file is not None):
            try:
                img = compress_image_bytes(uploaded_file.read(), max_dim=1024, quality=85)
                source_type = "🖥️ PC Upload"
            except Exception as e:
                st.error(f"Image processing error: {e}")
                st.stop()

    # ==== INPUT: File Upload ====
    else:
        st.markdown("### Upload Image from Device")
        uploaded_file = st.file_uploader("Select image file:", type=['jpg', 'jpeg', 'png'], key="direct_upload")
        if uploaded_file is not None:
            try:
                img = compress_image_bytes(uploaded_file.read(), max_dim=1024, quality=85)
                source_type = "📤 Uploaded File"
            except Exception as e:
                st.error(f"Image processing error: {e}")
                st.stop()

    # If no image, show message and wait
    if img is None:
        st.info("⏳ Waiting for image input... Select a photo to proceed.")
        st.stop()

    # Preview image
    st.markdown("#### Input Image Preview")
    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), width=300, caption=source_type)

    # ===== PHASE 1: Leaf Segmentation (YOLO) =====
    with st.spinner("PHASE 1: Segmenting leaf (YOLO)..."):
        try:
            yolo_results = yolo_model(img)
        except Exception as e:
            st.error(f"YOLO error: {e}")
            st.stop()
        if (not hasattr(yolo_results[0], "masks")) or (yolo_results[0].masks is None):
            st.error("❌ No leaf detected. Try with a clearer photo.")
            st.stop()
        leaf_mask = yolo_results[0].masks.data.cpu().numpy()[0]
    st.success("✅ Phase 1 Complete: Leaf Segmented")

    # ===== PHASE 2: Disease Segmentation (UNet) =====
    with st.spinner("PHASE 2: Segmenting disease areas (UNet)..."):
        img_tensor = preprocess_image(img).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            output = unet_model(img_tensor)
            pred_prob = torch.sigmoid(output)
            pred_mask = (pred_prob > 0.5).float().cpu().squeeze().numpy()
    st.success("✅ Phase 2 Complete: Disease Identified")

    # ===== PHASE 3: Severity Calculation + Disease Classification =====
    severity, severity_label, diseased_px, total_px = calculate_severity(pred_mask, leaf_mask)
    st.markdown(f"<h2 style='color:#d84315;'>📊 Severity: {severity:.2f}% ({severity_label})</h2>", unsafe_allow_html=True)

    disease_class, pred_confidence = "Unknown", 0.0
    if densenet_model is not None and class_names is not None:
        with st.spinner("PHASE 3: Classifying disease (DenseNet-121)..."):
            IMG_SIZE = 224
            img_tf = tf.image.resize_with_pad(tf.image.decode_image(cv2.imencode('.jpg', img)[1].tobytes(), channels=3), IMG_SIZE, IMG_SIZE)
            img_tf = tf.cast(img_tf, tf.float32)
            from tensorflow.keras.applications.densenet import preprocess_input
            img_tf = preprocess_input(img_tf)
            img_tf = tf.expand_dims(img_tf, 0)
            pred = densenet_model.predict(img_tf, verbose=0)
            pred_class = np.argmax(pred, axis=1)[0]
            disease_class = class_names[pred_class]
            pred_confidence = float(np.max(pred))
        st.markdown(f"<h3 style='color:#388e3c;'>🦠 Disease: {disease_class.replace('___', ' - ')}</h3>", unsafe_allow_html=True)
        st.markdown(f"<b>🎯 Model Confidence:</b> {pred_confidence:.2%}")
        st.success("✅ Phase 3 Complete: Disease Classified")
    else:
        st.warning("⚠️ DenseNet not loaded — classification skipped.")

    # ===== PHASE 4: Expert System Inference =====
    with st.spinner("PHASE 4: Expert System Inference (Knowledge Base)..."):
        expert_advice = phase4_expert_system_inference(disease_class, severity_label, knowledge_base)
    if "error" in expert_advice:
        st.warning(expert_advice["error"])
        st.stop()
    st.success("✅ Phase 4 Complete: Expert System Inference")

    # ===== PHASE 5: RAG + FAISS + Gemini =====
    with st.spinner("PHASE 5: RAG Pipeline (FAISS + LLM)..."):
        rag_context = phase5_rag_with_faiss(expert_advice, faiss_index, faiss_metadata, encoder, top_k=2)
        gemini_summary = generate_gemini_recommendation_phase5(expert_advice, rag_context, lang)
    st.success("✅ Phase 5 Complete: RAG + LLM Report Generated")

    # Display recommendations
    st.header("🌱 AI-Generated Recommendation")
    st.markdown(gemini_summary)

    # Detailed expert report
    with st.expander("📋 Full Expert System Report"):
        st.markdown(f"**Disease:** {expert_advice['disease_name']}")
        st.markdown(f"**Severity Level:** {expert_advice['severity_level']}")
        st.markdown(f"**Symptoms:** {expert_advice['symptoms']}")
        st.markdown(f"**Severity Description:** {expert_advice['severity_description']}")
        st.markdown("**Pesticides Recommended:**")
        for p in expert_advice['pesticides'].split(","):
            st.markdown(f"- {p.strip()}")
        st.markdown(f"**Treatment Plan:** {expert_advice['treatment']}")
        st.markdown("**Prevention Measures:**")
        for idx, line in enumerate(expert_advice['prevention'].split(". "), 1):
            if line.strip():
                st.markdown(f"{idx}. {line.strip()}.")

    # RAG pipeline details
    with st.expander("🔍 RAG Pipeline Details"):
        st.markdown("### FAISS Retrieved Contexts")
        for i, ctx in enumerate(rag_context['faiss_retrieved'], 1):
            st.markdown(f"**{i}. {ctx.get('disease_readable', 'N/A')}**")
            st.write(f"- Symptoms: {ctx.get('symptoms', 'N/A')}")
            st.write(f"- Pesticides: {ctx.get('pesticides', 'N/A')}")

    # Visualization
    st.header("📊 Segmentation Visualization")
    fig, axes = plt.subplots(1, 4, figsize=(18, 6))
    image_vis = cv2.cvtColor(cv2.resize(img, (pred_mask.shape[1], pred_mask.shape[0])), cv2.COLOR_BGR2RGB)
    axes[0].imshow(image_vis); axes[0].set_title("Original Image"); axes[0].axis('off')
    axes[1].imshow(leaf_mask, cmap='Greens', alpha=0.8); axes[1].set_title("Leaf Mask (YOLO)"); axes[1].axis('off')
    axes[2].imshow(pred_mask, cmap='autumn', alpha=0.8); axes[2].set_title("Disease Mask (UNet)"); axes[2].axis('off')
    overlay = np.zeros_like(pred_mask)
    overlay[(pred_mask > 0) & (resize(leaf_mask, pred_mask.shape, order=0, preserve_range=True, anti_aliasing=False) > 0)] = 1
    axes[3].imshow(image_vis)
    axes[3].imshow(overlay, cmap='autumn', alpha=0.5)
    axes[3].set_title(f"Disease Overlay\n{disease_class.replace('___', ' - ')}\n{severity:.2f}% ({severity_label})")
    axes[3].axis('off')
    st.pyplot(fig)

    # JSON report download
    results = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source": source_type,
        "phase_1_segmentation": "Leaf segmented (YOLOv8)",
        "phase_2_disease_segmentation": "Disease area identified (U-Net)",
        "phase_3_classification": {
            "disease": disease_class,
            "confidence": pred_confidence,
            "severity_percent": severity,
            "severity_label": severity_label
        },
        "phase_4_expert_inference": expert_advice,
        "phase_5_rag_output": {
            "faiss_contexts_retrieved": len(rag_context['faiss_retrieved']),
            "gemini_summary": gemini_summary,
            "language": lang
        },
    }
    st.download_button(
        "⬇️ Download Full Report (JSON)",
        json.dumps(results, indent=2),
        file_name="leafsense_analysis_report.json",
        mime="application/json"
    )


if __name__ == "__main__":
    main()
