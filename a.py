import streamlit as st
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
import warnings
warnings.filterwarnings('ignore')
from datetime import datetime

st.set_page_config(page_title="LeafSense: Disease Detection & Expert System", layout="wide", initial_sidebar_state="expanded")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- CACHE LOADERS ---
@st.cache_resource(show_spinner=True)
def load_yolo_model(path):
    return YOLO(path)

@st.cache_resource(show_spinner=True)
def load_unet_model(path):
    model = smp.from_pretrained(path)
    model.to(DEVICE).eval()
    return model

@st.cache_resource(show_spinner=True)
def load_densenet_model(path):
    class DummyCast(tf.keras.layers.Layer):
        def __init__(self, dtype=None, **kwargs):
            super().__init__(**kwargs)
            self._dtype = dtype
        def call(self, inputs):
            return inputs
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
def load_class_names(path):
    return np.load(path)

@st.cache_data(show_spinner=True)
def load_knowledge_base(path):
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except Exception:
        return {}

# --- NORMALIZATION ---
def normalize_class_name(name):
    return (name.replace(" ", "_")
                .replace("-", "_")
                .replace("__", "___")
                .strip()
                .lower()
    )

# --- EXPERT QUERY WITH NORMALIZATION ---
def query_expert_system(disease_class, severity_label, knowledge_base):
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

# --- RAG DOCUMENT CREATION ---
def create_rag_documents(knowledge_base):
    documents = []
    for disease_name, disease_info in knowledge_base.items():
        doc_content = f"""
Disease: {disease_name.replace('___', ' - ').replace('_', ' ')}

SYMPTOMS:
{disease_info['symptoms']}

PESTICIDES:
{disease_info['pesticides']}

PREVENTION MEASURES:
{disease_info['prevention']}

MILD SEVERITY ({disease_info['severity_identification']['mild']}):
Treatment: {disease_info['solutions']['mild']}

MODERATE SEVERITY ({disease_info['severity_identification']['moderate']}):
Treatment: {disease_info['solutions']['moderate']}

SEVERE SEVERITY ({disease_info['severity_identification']['severe']}):
Treatment: {disease_info['solutions']['severe']}
"""
        documents.append({
            "disease": disease_name,
            "content": doc_content,
            "metadata": {
                "disease_readable": disease_name.replace("___", " - ").replace("_", " "),
                "pesticides": disease_info["pesticides"].split(", "),
                "severity_levels": list(disease_info["severity_identification"].keys())
            }
        })
    return documents

def simple_retrieval(query, documents, top_k=3):
    query_lower = query.lower()
    scored_docs = []
    for doc in documents:
        score = 0
        content_lower = doc["content"].lower()
        if doc["disease"].lower() in query_lower or query_lower in doc["disease"].lower():
            score += 10
        keywords = query_lower.split()
        for keyword in keywords:
            if len(keyword) > 2 and keyword in content_lower:
                score += 1
        if score > 0:
            scored_docs.append((doc, score))
    scored_docs.sort(key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in scored_docs[:top_k]]

def rag_query_with_context(
    disease_class, severity_label, user_query, rag_documents, knowledge_base
):
    expert_advice = query_expert_system(
        disease_class=disease_class,
        severity_label=severity_label,
        knowledge_base=knowledge_base
    )
    combined_query = f"{disease_class} {severity_label} {user_query}" if user_query else f"{disease_class} {severity_label}"
    retrieved_docs = simple_retrieval(combined_query, rag_documents, top_k=2)
    rag_enhanced_result = {
        "expert_system_output": expert_advice,
        "rag_retrieved_documents": retrieved_docs,
        "retrieved_context": [
            {
                "disease": doc["disease"],
                "disease_readable": doc["metadata"]["disease_readable"],
                "relevance": "primary" if doc["disease"] == disease_class else "supporting",
                "pesticides": doc["metadata"]["pesticides"]
            }
            for doc in retrieved_docs
        ],
        "retrieval_summary": {
            "total_documents_retrieved": len(retrieved_docs),
            "primary_match": retrieved_docs[0]["metadata"]["disease_readable"] if retrieved_docs else "None",
            "confidence": "High" if retrieved_docs else "Medium"
        }
    }
    return rag_enhanced_result

# --- IMAGE PREPROCESSING ---
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
    if severity < 10:
        label = "Mild"
    elif severity < 30:
        label = "Moderate"
    else:
        label = "Severe"
    return severity, label, diseased_pixels, total_pixels

# ----------- STREAMLIT MAIN UI -----------

def main():
    st.markdown(
        "<h1 style='text-align: center; color: #26a69a;'>LeafSense Disease Detection & Expert System</h1>",
        unsafe_allow_html=True
    )
    st.sidebar.markdown("## Instructions")
    st.sidebar.write(
        "1. Upload a leaf image or take a photo.\n"
        "2. Segmentation and disease detection will run automatically.\n"
        "3. Results and expert recommendations will appear below."
    )

    st.markdown("#### Upload an image or take a photo (File Upload best for desktop, Camera for mobile)")
    col1, col2 = st.columns([1,1])
    with col1:
        uploaded_file = st.file_uploader("Upload Leaf Image", type=['jpg', 'jpeg', 'png'])
    with col2:
        captured_image = st.camera_input("Take Leaf Photo (Mobile)")

    if uploaded_file:
        input_image_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(input_image_bytes, cv2.IMREAD_COLOR)
        source_type = "Uploaded file"
    elif captured_image:
        img = cv2.imdecode(np.frombuffer(captured_image.getvalue(), np.uint8), cv2.IMREAD_COLOR)
        source_type = "Captured photo"
    else:
        st.info("Please upload or capture a leaf image to begin.")
        return

    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption=f"{source_type}", use_column_width=True)

    # --- Load models/data
    with st.spinner("Loading models..."):
        yolo_model = load_yolo_model("leafsense_best.pt")
        unet_model = load_unet_model("best_weights.pth")
        densenet_model = load_densenet_model("densenet121_final_model.h5")
        class_names = load_class_names("class_names.npy")
        knowledge_base = load_knowledge_base("expert_knowledge_base.json")
        rag_documents = create_rag_documents(knowledge_base)

    st.sidebar.success(f"DenseNet loaded: {densenet_model is not None}")
    st.sidebar.success(f"Classes loaded: {isinstance(class_names, np.ndarray)}")
    st.sidebar.write(f"Sample classes: {class_names[:5]}")

    # PHASE 1: Leaf Segmentation (YOLO)
    with st.spinner("Leaf segmentation with YOLO..."):
        yolo_results = yolo_model(img)
        if (not hasattr(yolo_results[0], "masks")) or (yolo_results[0].masks is None):
            st.error("No leaf detected in the image. Please try again with a clearer leaf photo!")
            return
        leaf_mask = yolo_results[0].masks.data.cpu().numpy()[0]
    st.success("Leaf segmentation complete!")

    # PHASE 2: Disease Segmentation (UNet)
    with st.spinner("Disease segmentation (UNet)..."):
        img_tensor = preprocess_image(img).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            output = unet_model(img_tensor)
            pred_prob = torch.sigmoid(output)
            pred_mask = (pred_prob > 0.5).float().cpu().squeeze().numpy()
    st.success("Disease segmentation complete!")

    # Severity calculation
    severity, severity_label, diseased_px, total_px = calculate_severity(pred_mask, leaf_mask)
    st.markdown(f"<h2 style='color:#d84315;'>Disease Severity: {severity:.2f}% ({severity_label})</h2>", unsafe_allow_html=True)

    # PHASE 3: Disease Classification
    disease_class, pred_confidence = "Unknown", 0.0
    if densenet_model is not None and class_names is not None:
        with st.spinner("Classifying disease (DenseNet121)..."):
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
    else:
        st.warning("DenseNet model or class names not loaded. Classification skipped.")

    # PHASE 4: Expert System + RAG Context
    st.header("🌱 Expert System & RAG Context Recommendations")
    rag_result = rag_query_with_context(
        disease_class=disease_class,
        severity_label=severity_label,
        user_query="",  # you can prompt for a user query if desired
        rag_documents=rag_documents,
        knowledge_base=knowledge_base
    )
    advice = rag_result["expert_system_output"]

    if "error" in advice:
        st.warning(advice["error"])
        st.markdown("Closest match in KB: " + ", ".join([doc["disease_readable"] for doc in rag_result["rag_retrieved_documents"]]))
    else:
        st.markdown(f"**Disease:** {advice['disease_name']}")
        st.markdown(f"**Severity Level:** {advice['severity_level']}")
        st.markdown(f"**Symptoms:** {advice['symptoms']}")
        st.markdown(f"**Severity Description:** {advice['severity_description']}")
        st.markdown("**Pesticides Recommended:**")
        for p in advice['pesticides'].split(","):
            st.markdown(f"- {p.strip()}")
        st.markdown(f"**Treatment:** {advice['treatment']}")
        st.markdown("**Prevention Measures:**")
        for idx, line in enumerate(advice['prevention'].split(". ")):
            if line.strip():
                st.markdown(f"{idx+1}. {line.strip()}.")

    # Show RAG context (retrieved docs) for reference
    with st.expander("🔍 Retrieved RAG Context"):
        for doc in rag_result["rag_retrieved_documents"]:
            st.markdown(f"**{doc['metadata']['disease_readable']}**")
            st.markdown(doc["content"])

    # PHASE 5: Visualization
    st.header("📊 Segmentation Visualization")
    fig, axes = plt.subplots(1, 4, figsize=(18, 6))
    image_vis = cv2.cvtColor(cv2.resize(img, (pred_mask.shape[1], pred_mask.shape[0])), cv2.COLOR_BGR2RGB)
    axes[0].imshow(image_vis)
    axes[0].set_title("Original Image")
    axes[0].axis('off')
    axes[1].imshow(leaf_mask, cmap='Greens', alpha=0.8)
    axes[1].set_title("Leaf Mask (YOLO)")
    axes[1].axis('off')
    axes[2].imshow(pred_mask, cmap='autumn', alpha=0.8)
    axes[2].set_title("Disease Mask (UNet)")
    axes[2].axis('off')
    overlay = np.zeros_like(pred_mask)
    overlay[(pred_mask > 0) & (resize(leaf_mask, pred_mask.shape, order=0, preserve_range=True, anti_aliasing=False) > 0)] = 1
    axes[3].imshow(image_vis)
    axes[3].imshow(overlay, cmap='autumn', alpha=0.5)
    axes[3].set_title(f"Overlay & Prediction\n{disease_class.replace('___', ' - ')}\n{severity:.2f}% ({severity_label})")
    axes[3].axis('off')
    st.pyplot(fig)

    # Save all outputs in a download-ready JSON report (optional)
    results = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "disease_class": disease_class,
        "severity_label": severity_label,
        "severity_percentage": severity,
        "confidence": pred_confidence,
        "advice": advice,
        "rag_context": rag_result["retrieved_context"],
        "retrieval_summary": rag_result["retrieval_summary"]
    }
    json_out = json.dumps(results, indent=2)
    st.download_button("Download Full Analysis as JSON", json_out, file_name="leafsense_analysis_report.json", mime="application/json")


if __name__ == "__main__":
    main()
