import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import pickle
import os

def build_faiss_index(kb_path, output_index_path, output_metadata_path):
    """
    Build FAISS vector index from knowledge base.
    Stores vectors of disease descriptions + symptoms + prevention.
    
    Args:
        kb_path: Path to expert_knowledge_base.json
        output_index_path: Path to save faiss_index.bin
        output_metadata_path: Path to save faiss_metadata.pkl
    """
    print("📖 Loading Knowledge Base...")
    if not os.path.exists(kb_path):
        print(f"❌ Error: {kb_path} not found!")
        return
    
    with open(kb_path, 'r') as f:
        kb = json.load(f)
    print(f"✅ Loaded {len(kb)} diseases from KB")

    print("🔄 Loading Sentence Transformer (all-MiniLM-L6-v2)...")
    encoder = SentenceTransformer('all-MiniLM-L6-v2')
    print("✅ Encoder loaded")
    
    vectors = []
    metadata = []
    
    print("📝 Encoding disease descriptions...")
    for idx, (disease_name, disease_info) in enumerate(kb.items()):
        # Combine key info for embedding (semantic search)
        text = f"{disease_name} {disease_info['symptoms']} {disease_info['pesticides']} {disease_info['prevention']}"
        
        # Generate embedding
        vector = encoder.encode(text, convert_to_numpy=True)
        vectors.append(vector)
        
        metadata.append({
            "disease": disease_name,
            "disease_readable": disease_name.replace("___", " - ").replace("_", " "),
            "symptoms": disease_info['symptoms'],
            "pesticides": disease_info['pesticides'],
            "prevention": disease_info['prevention'],
            "severity_levels": list(disease_info.get('severity_identification', {}).keys()),
            "solutions": disease_info.get('solutions', {})
        })
        print(f"  [{idx+1}/{len(kb)}] {disease_name.replace('___', ' - ')} ✅")
    
    print(f"\n🔧 Building FAISS index...")
    # Convert to numpy array
    vectors_array = np.array(vectors).astype('float32')
    print(f"  Vector shape: {vectors_array.shape}")
    
    # Build FAISS index (L2 distance)
    dimension = vectors_array.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(vectors_array)
    print(f"  ✅ Index created with {index.ntotal} vectors, dimension: {dimension}")
    
    # Save index
    faiss.write_index(index, output_index_path)
    print(f"✅ FAISS Index saved: {output_index_path}")
    
    # Save metadata
    with open(output_metadata_path, 'wb') as f:
        pickle.dump(metadata, f)
    print(f"✅ Metadata saved: {output_metadata_path}")
    
    print("\n" + "="*60)
    print("📊 FAISS INDEX BUILD SUMMARY")
    print("="*60)
    print(f"Total diseases indexed: {len(vectors)}")
    print(f"Vector dimension: {dimension}")
    print(f"Index type: FAISS IndexFlatL2 (L2 distance)")
    print(f"Index file size: {os.path.getsize(output_index_path) / (1024*1024):.2f} MB")
    print(f"Metadata file size: {os.path.getsize(output_metadata_path) / 1024:.2f} KB")
    print("="*60)
    print("✅ Ready for Phase 5 RAG retrieval!")
    print("="*60)

if __name__ == "__main__":
    # Default paths (adjust if needed)
    KB_PATH = "expert_knowledge_base.json"
    INDEX_PATH = "faiss_index.bin"
    METADATA_PATH = "faiss_metadata.pkl"
    
    print("\n" + "="*60)
    print("🚀 FAISS INDEX BUILDER FOR LEAFSENSE")
    print("="*60)
    print(f"Knowledge Base: {KB_PATH}")
    print(f"Output Index: {INDEX_PATH}")
    print(f"Output Metadata: {METADATA_PATH}")
    print("="*60 + "\n")
    
    build_faiss_index(KB_PATH, INDEX_PATH, METADATA_PATH)
