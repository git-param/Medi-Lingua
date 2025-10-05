import streamlit as st
import pickle
import pandas as pd
from sentence_transformers import SentenceTransformer
import torch
import numpy as np
from src.search import init_faiss, init_tfidf

@st.cache_resource
def load_model():
    """Load the medical Sentence Transformer model (SapBERT)."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_name = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
    st.info(f"🔬 Loading model: {model_name} on {device.upper()}")
    model = SentenceTransformer(model_name, device=device)
    return model

@st.cache_resource
def load_data():
    """
    Loads SapBERT-based question embeddings and the corresponding original dataset.
    Also initializes FAISS and TF-IDF for fast hybrid search.
    """
    try:
        # --- Load embeddings ---
        with open('dataset/question_embeddings.pkl', 'rb') as f:
            embedding_data = pickle.load(f)
        embeddings = embedding_data.get('embeddings')
        embeddings = embeddings.astype('float32')  # FAISS requires float32
        num_embeddings = embeddings.shape[0]

        # --- Load original dataset ---
        original_df = pd.read_csv('dataset/dataset.csv')
        original_df.dropna(subset=['Description', 'Patient', 'Doctor'], inplace=True)
        original_df.drop_duplicates(inplace=True)

        # Trim to match embeddings
        original_df = original_df.iloc[:num_embeddings]
        original_answers = original_df['Doctor'].tolist()
        patient_column = original_df['Patient'].tolist()
        description_column = original_df['Description'].tolist()

        # --- Initialize hybrid search ---
        corpus_texts = [p + " " + d for p, d in zip(patient_column, description_column)]
        init_tfidf(corpus_texts)
        init_faiss(embeddings)

        # --- Runtime checks ---
        st.write(f"✅ Loaded embeddings: {embeddings.shape}")
        if embeddings.shape[1] != 768:
            st.warning(
                f"⚠️ Embedding dimension is {embeddings.shape[1]}, expected 768 for SapBERT."
            )

        return {
            "embeddings": embeddings,
            "original_answers": original_answers,
            "patient_column": patient_column,
            "description_column": description_column
        }

    except FileNotFoundError as e:
        st.error(f"❌ Data file not found. Details: {e}")
        return None
    except Exception as e:
        st.error(f"❌ Error while loading data: {e}")
        return None
