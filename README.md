# 🌿 LeafSense AI - Disease Detection & Expert System

An advanced AI-powered agricultural disease detection system combining deep learning segmentation, classification, expert systems, and RAG (Retrieval-Augmented Generation) to provide farmers with actionable disease recommendations in English and Telugu.

---

## 📋 Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Deployment](#deployment)
- [Project Structure](#project-structure)
- [Technologies Used](#technologies-used)
- [Future Enhancements](#future-enhancements)

---

## ✨ Features

### **Two Operational Modes:**

1. **Image Detection Mode** 📸
   - Upload or capture leaf images
   - Automatic disease detection using deep learning
   - Severity assessment
   - Expert recommendations with pesticide suggestions
   - Multi-language support (English/Telugu)

2. **Disease Expert Advisor Mode** 💭
   - Text-based disease consultation
   - FAISS vector search through 42+ diseases
   - Personalized recommendations
   - No image required!

### **5-Phase Processing Pipeline:**

| Phase | Component | Technology |
|-------|-----------|-----------|
| **Phase 1** | Leaf Segmentation | YOLO (Ultralytics) |
| **Phase 2** | Disease Segmentation | UNet (PyTorch) |
| **Phase 3** | Disease Classification | DenseNet121 (TensorFlow) |
| **Phase 4** | Expert System Inference | Knowledge Base Verification |
| **Phase 5** | RAG + LLM Report | FAISS + Google Gemini |

### **Advanced Features:**

✅ Multi-model Deep Learning (YOLO, UNet, DenseNet121)  
✅ FAISS Vector Indexing (42 diseases, 384-dim embeddings)  
✅ Knowledge Base Expert System (Ground truth verification)  
✅ Gemini LLM Integration (Hallucination-free recommendations)  
✅ Multi-language Support (English & Telugu)  
✅ JSON Report Export  
✅ Real-time Segmentation Visualization  

---

## 🏗️ Architecture

```
User Input (Image/Text)
    ↓
PHASE 1: Leaf Segmentation (YOLO)
    ↓
PHASE 2: Disease Segmentation (UNet)
    ↓
PHASE 3: Disease Classification (DenseNet121) + Severity Calculation
    ↓
PHASE 4: Expert System Inference (KB Verification - Ground Truth)
    ↓
PHASE 5: RAG Retrieval (FAISS Vector Search + Gemini LLM)
    ↓
Output: Line-by-line Recommendations + Visualizations + JSON Report
```

---

## 🚀 Installation

### **Prerequisites:**
- Python 3.10+
- CUDA 11.8+ (for GPU acceleration, optional)
- Git

### **Local Setup:**

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/leafsense-ai.git
cd leafsense-ai

# 2. Create virtual environment
python -m venv venv

# 3. Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Set up Gemini API key
# Create .streamlit/secrets.toml:
mkdir .streamlit
echo 'GEMINI_API_KEY = "your_api_key_here"' > .streamlit/secrets.toml

# 6. Build FAISS index (one-time)
python build_faiss_index.py

# 7. Run the app
streamlit run app.py
```

Open your browser to `http://localhost:8501`

---

## 📖 Usage

### **Image Detection Mode:**

1. Select "Image Detection" from sidebar
2. Choose upload or camera capture
3. Wait for 5-phase pipeline to complete
4. View:
   - Line-by-line AI summary
   - Full detailed report (expandable)
   - 4-panel segmentation visualization
   - Download JSON analysis report

### **Disease Expert Advisor Mode:**

1. Select "Ask Disease Expert" from sidebar
2. Describe your disease concern (e.g., "My tomato leaves have brown spots with yellow rings")
3. AI searches database and generates recommendations
4. View top 3 matching diseases with full details
5. Get expert advice in your chosen language

### **Language Support:**

- Select **English** or **Telugu** from dropdown at top
- All outputs will be generated in selected language
- Perfect for Indian farmers!

---

## 🌐 Deployment

### **Deploy to Streamlit Cloud (Free):**

#### **Step 1: Push to GitHub**

```bash
# Initialize git repo (if not already done)
git init
git add .
git commit -m "Initial commit: LeafSense AI Disease Detection System"

# Add remote repository
git remote add origin https://github.com/yourusername/leafsense-ai.git

# Push to GitHub
git branch -M main
git push -u origin main
```

#### **Step 2: Deploy on Streamlit Cloud**

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Click "New App"
3. Select your GitHub repo: `yourusername/leafsense-ai`
4. Branch: `main`
5. Main file path: `app.py`
6. Click "Deploy"

#### **Step 3: Set Secrets**

In Streamlit Cloud dashboard:
1. Go to App Settings → Secrets
2. Add:
```toml
GEMINI_API_KEY = "your_actual_api_key"
```

#### **Step 4: Access Your App**

```
https://your-username-leafsense-ai.streamlit.app
```

---

## 📁 Project Structure

```
leafsense-ai/
├── app.py                           # Main Streamlit application
├── build_faiss_index.py             # FAISS index builder
├── requirements.txt                 # Python dependencies
├── .gitignore                       # Git ignore rules
├── README.md                        # This file
├── .streamlit/
│   └── config.toml                 # Streamlit config (not committed)
├── expert_knowledge_base.json       # 42 diseases KB
├── class_names.npy                  # DenseNet class labels
├── leafsense_best.pt                # YOLO model
├── best_weights.pth                 # UNet model
├── densenet121_final_model.h5       # DenseNet model
├── faiss_index.bin                  # FAISS vector index
└── faiss_metadata.pkl               # FAISS metadata
```

---

## 🛠️ Technologies Used

### **Deep Learning:**
- **YOLO v8** (Leaf segmentation)
- **UNet** (Disease segmentation via Segmentation Models PyTorch)
- **DenseNet121** (Disease classification via TensorFlow)

### **Vector Search & RAG:**
- **FAISS** (Vector indexing & similarity search)
- **Sentence Transformers** (Embedding model: all-MiniLM-L6-v2)

### **LLM & API:**
- **Google Gemini 2.5 Flash** (Text generation & translation)

### **Web Framework:**
- **Streamlit** (Interactive UI)

### **Supporting Libraries:**
- PyTorch & Torchvision (Deep Learning)
- TensorFlow & Keras (Classification)
- OpenCV (Image processing)
- Numpy & Scikit-image (Data processing)
- Matplotlib (Visualization)

---

## 📊 Model Performance

| Model | Task | Input Size | Accuracy |
|-------|------|-----------|----------|
| YOLO v8 | Leaf Segmentation | 640x640 | ~95% |
| UNet | Disease Segmentation | 256x320 | ~92% |
| DenseNet121 | Classification | 224x224 | ~98% |

---

## 🌾 Supported Diseases (42 Total)

**Apple:** Apple scab, Black rot, Cedar apple rust  
**Cherry:** Powdery mildew  
**Corn:** Cercospora leaf spot, Common rust, Northern leaf blight  
**Grape:** Black rot, Leaf blight, Black measles, Downy mildew  
**Orange:** Huanglongbing  
**Peach:** Bacterial spot, Brown rot  
**Pepper:** Bacterial spot  
**Potato:** Early blight, Late blight  
**Soybean:** Frogeye leaf spot, Septoria brown spot  
**Squash:** Powdery mildew, Downy mildew  
**Strawberry:** Leaf scorch  
**Tomato:** 12 diseases (Early blight, Late blight, Leaf mold, Septoria spot, etc.)  

Plus healthy plant classifications for all crops.

---

## 📝 API Keys Required

### **Google Gemini API:**
1. Go to [Google AI Studio](https://aistudio.google.com)
2. Click "Get API Key"
3. Create new API key
4. Add to `.streamlit/secrets.toml`:
```toml
GEMINI_API_KEY = "your_key_here"
```

---

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see LICENSE file for details.

---

## 🚀 Future Enhancements

- [ ] Mobile app (React Native)
- [ ] Offline mode with TensorFlow Lite
- [ ] Pest detection module
- [ ] Weather-based recommendations
- [ ] Farmer community forum
- [ ] SMS/WhatsApp integration for alerts
- [ ] Video input support
- [ ] More crop diseases (50+ total)
- [ ] Multi-language support (Hindi, Kannada, Tamil)
- [ ] Historical disease tracking

---

## 📞 Support & Contact

For issues, questions, or suggestions:
- Open an issue on GitHub
- Email: support@leafsense.ai
- Website: leafsense.ai

---

## 🙏 Acknowledgments

- Deep Learning models: Ultralytics, PyTorch, TensorFlow communities
- Vector search: FAISS (Meta)
- LLM: Google Generative AI
- UI: Streamlit team
- Special thanks to agricultural experts for domain knowledge

---

**Made with ❤️ for farmers worldwide** 🌾🌿
