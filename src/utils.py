import streamlit as st
import pickle
import pandas as pd
from sentence_transformers import SentenceTransformer, models
import torch
import numpy as np
from src.search import init_faiss
import os

BASE_DIR = os.path.dirname(__file__)  # src/
QUESTION_EMB_PATH = os.path.join(BASE_DIR, "..", "dataset", "question_embeddings.pkl")
DOCTOR_EMB_PATH = os.path.join(BASE_DIR, "..", "dataset", "doctor_embeddings.pkl")
DATASET_CSV_PATH = os.path.join(BASE_DIR, "..", "dataset", "dataset.csv")


@st.cache_resource
def load_model():
    """Load the SapBERT model properly from a local folder."""
    import torch
    from transformers import AutoTokenizer, AutoModel
    from sentence_transformers import SentenceTransformer, models

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 👇 Local model path instead of downloading from Hugging Face
    model_name = "models/SapBERT-from-PubMedBERT-fulltext"
    st.info(f"🔬 Loading SapBERT locally from '{model_name}' on {device.upper()}...")

    try:
        # Load the Transformer + Pooling manually for full control
        word_embedding_model = models.Transformer(model_name)
        pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
        model = SentenceTransformer(modules=[word_embedding_model, pooling_model], device=device)
        st.success("✅ SapBERT (local) loaded successfully.")
    except Exception as e:
        st.error(f"❌ Failed to load SapBERT locally. Details: {e}")
        st.warning("⚠️ Falling back to 'all-MiniLM-L6-v2' model (general-purpose).")
        model = SentenceTransformer('all-MiniLM-L6-v2', device=device)

    return model


@st.cache_resource
def load_data():
    """
    Loads embeddings and the corresponding dataset.
    Returns a dictionary with description+patient embeddings, doctor embeddings, and text columns.
    """
    try:
        # --- Load question embeddings ---
        
        with open(QUESTION_EMB_PATH, 'rb') as f:
            question_data = pickle.load(f)
        question_embeddings = question_data.get('embeddings').astype('float32')

        # --- Load doctor embeddings ---
        with open(DOCTOR_EMB_PATH, 'rb') as f:
            doctor_data = pickle.load(f)
        doctor_embeddings = doctor_data.get('embeddings').astype('float32')

        # --- Load dataset ---
        df = pd.read_csv('dataset/dataset.csv')
        df.dropna(subset=['Description', 'Patient', 'Doctor'], inplace=True)
        df.drop_duplicates(inplace=True)

        # Ensure all arrays align
        num_samples = min(len(df), len(question_embeddings), len(doctor_embeddings))
        df = df.iloc[:num_samples]
        question_embeddings = question_embeddings[:num_samples]
        doctor_embeddings = doctor_embeddings[:num_samples]

        # No normalization — already normalized before saving
        st.success(f"✅ Loaded {num_samples} rows with SapBERT embeddings ({question_embeddings.shape[1]} dims)")

        # Initialize FAISS with question embeddings for retrieval
        init_faiss(question_embeddings)

        return {
            "question_embeddings": question_embeddings,
            "doctor_embeddings": doctor_embeddings,
            "description_column": df["Description"].tolist(),
            "patient_column": df["Patient"].tolist(),
            "original_answers": df["Doctor"].tolist(),
        }

    except FileNotFoundError as e:
        st.error(f"❌ Embedding/data file not found. Details: {e}")
        return None
    except Exception as e:
        st.error(f"❌ Error while loading data: {e}")
        return None
