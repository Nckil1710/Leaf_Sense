# app.py (updated: mobile-first capture + immediate preview)
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

# ---------- Utility & pipeline functions ----------
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

# Optional: small switchable widget HTML (kept as fallback)
CAMERA_WIDGET = r"""
<style>
  .btn { padding:6px 10px; border-radius:6px; background:#1976d2; color:white; border:none; font-weight:600; cursor:pointer; margin-right:6px; }
</style>
<div>
  <div style="margin-bottom:8px;">
    <button id="startBtn" class="btn">Start Camera (prefers back)</button>
    <button id="switchBtn" class="btn">Switch Camera</button>
    <button id="snapBtn" class="btn">Capture & Send</button>
  </div>
  <video id="video" width="320" height="240" autoplay playsinline style="border:1px solid #ddd;"></video>
  <canvas id="canvas" width="320" height="240" style="display:none;"></canvas>
</div>
<script>
let currentStream = null;
async function startStream(constraints) {
  if (currentStream) currentStream.getTracks().forEach(t=>t.stop());
  try {
    currentStream = await navigator.mediaDevices.getUserMedia(constraints);
    document.getElementById('video').srcObject = currentStream;
  } catch(e){ alert('Camera error: '+e.message); }
}
document.getElementById('startBtn').onclick = ()=> startStream({video:{facingMode:'environment'}, audio:false});
document.getElementById('switchBtn').onclick = async ()=>{
  // try toggling facingMode
  await startStream({video:{facingMode:'user'}, audio:false});
};
document.getElementById('snapBtn').onclick = ()=>{
  const v = document.getElementById('video'), c=document.getElementById('canvas');
  c.width = v.videoWidth; c.height = v.videoHeight;
  c.getContext('2d').drawImage(v,0,0);
  const data = c.toDataURL('image/jpeg',0.9);
  // send to parent via postMessage (some browsers)
  window.parent.postMessage({isStreamlitImage:true, data}, "*");
  // also write to title for fallback
  document.title = data;
  alert('Captured — now close this dialog and paste the data URL into the "Paste data URL" box OR use the Upload fallback.');
};
</script>
"""

def dataurl_to_cv2_img(data_url):
    header, encoded = data_url.split(',', 1)
    data = base64.b64decode(encoded)
    image = Image.open(BytesIO(data)).convert("RGB")
    open_cv_image = np.array(image)[:, :, ::-1]  # RGB->BGR
    return open_cv_image

def main():
    st.markdown("<h1 style='text-align: center; color:#1976d2;'>🌿 LeafSense Disease Detection & Expert System</h1>", unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.markdown("## 🌱 Welcome to LeafSense!")
        mode = st.radio("Mode:", ["Image Detection", "Ask Disease Expert"])
        st.write("---")
        if mode == "Image Detection":
            st.write("Upload / Capture a clear leaf photo. On mobile, choose the Camera option in the picker.")
        else:
            st.write("Describe symptoms to get expert advice.")

    lang = st.selectbox("Choose Output Language:", ["English", "Telugu"])

    # Load models
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
        st.markdown("Tap the **Upload / Camera** button below on mobile to open camera. If you prefer, use the optional in-page camera widget (fallback).")

        # Primary: use Streamlit's file_uploader (mobile browsers typically show camera option here)
        uploaded_file = st.file_uploader("Capture or upload image", type=['jpg','jpeg','png'], accept_multiple_files=False, help="On mobile choose Camera to capture photo.")
        img = None
        source_type = None

        # If user used file_uploader (most common mobile path), show preview immediately
        if uploaded_file is not None:
            input_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            img = cv2.imdecode(input_bytes, cv2.IMREAD_COLOR)
            source_type = "Uploaded / Captured via file picker"

        # Optional: show the small switchable camera widget
        with st.expander("Use in-page camera widget (optional)"):
            st.markdown("This widget can prefer back camera and lets you switch; captured image must be pasted or uploaded if automatic transfer fails.")
            components.html(CAMERA_WIDGET, height=340)
            pasted = st.text_area("If you used the widget and the image didn't auto-import, paste the captured Data URL here (starts with 'data:image/jpeg;base64,')", height=80)
            if pasted and pasted.strip().startswith("data:image"):
                try:
                    img = dataurl_to_cv2_img(pasted.strip())
                    source_type = "Captured (widget pasted)"
                except Exception:
                    st.error("Failed to decode pasted data URL.")

        if img is None:
            st.info("Please capture or upload a leaf image to begin.")
            st.stop()

        # Show preview
        st.markdown("#### Input image preview")
        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), use_column_width=False, width=360, caption=source_type)

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

        # Detailed report & visualizations
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
        axes[3].set_title(f"Overlay & Prediction\n{disease_class.replace('___', ' - ')}\n{severity:.2f}% ({severity_label})")
        axes[3].axis('off')
        st.pyplot(fig)

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
        disease_query = st.text_area("Describe disease/symptoms:", placeholder="e.g., 'tomato brown spots with yellow rings'", height=100)
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
