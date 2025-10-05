import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import faiss  # pip install faiss-cpu
import re


# Global caches
tfidf_vectorizer = None
tfidf_matrix = None
corpus_texts = None
faiss_index = None
embeddings_array = None  # FAISS needs float32

def encode_question(model, user_question):
    """Encodes the user's question using the embedding model."""
    if model is None or not user_question.strip():
        return None
    return model.encode([user_question], show_progress_bar=False)[0].astype('float32')

def init_tfidf(data_texts):
    """
    Initialize TF-IDF matrix for hybrid search.
    """
    global tfidf_vectorizer, tfidf_matrix, corpus_texts
    corpus_texts = data_texts
    tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=10000)
    tfidf_matrix = tfidf_vectorizer.fit_transform(corpus_texts)

def init_faiss(embeddings):
    """
    Initialize FAISS index for fast semantic search.
    embeddings: np.array (num_samples x 768) normalized
    """
    global faiss_index, embeddings_array
    embeddings_array = embeddings.astype('float32')
    dimension = embeddings_array.shape[1]
    faiss_index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
    faiss_index.add(embeddings_array)

def find_top_matches_indices(model, embeddings, user_question, top_k=3, weight_semantic=0.7, faiss_top_candidates=100):
    """
    Hybrid top-k search:
    - weight_semantic: 0-1
    - FAISS used for fast semantic search
    - TF-IDF applied only on top semantic candidates
    """
    global tfidf_vectorizer, tfidf_matrix, corpus_texts, faiss_index, embeddings_array

    if model is None or embeddings is None or len(embeddings) == 0:
        return []

    question_embedding = encode_question(model, user_question)
    if question_embedding is None:
        return []

    # --- 1. Fast semantic search with FAISS ---
    if faiss_index is not None:
        D, I = faiss_index.search(np.array([question_embedding]), k=min(faiss_top_candidates, embeddings.shape[0]))
        top_candidates = I[0]
        semantic_sim_top = D[0]
    else:
        # fallback: full cosine similarity
        semantic_sim_full = np.dot(embeddings, question_embedding)
        top_candidates = np.argpartition(semantic_sim_full, -faiss_top_candidates)[-faiss_top_candidates:]
        top_candidates = top_candidates[np.argsort(semantic_sim_full[top_candidates])[::-1]]
        semantic_sim_top = semantic_sim_full[top_candidates]

    # --- 2. TF-IDF similarity on top candidates only ---
    if tfidf_vectorizer is not None and tfidf_matrix is not None:
        tfidf_vec = tfidf_vectorizer.transform([user_question])
        tfidf_sim_top = (tfidf_matrix[top_candidates] @ tfidf_vec.T).toarray().ravel()
    else:
        tfidf_sim_top = np.zeros(len(top_candidates))

    # --- 3. Combine similarities ---
    combined_sim_top = weight_semantic * semantic_sim_top + (1 - weight_semantic) * tfidf_sim_top

    # --- 4. Select final top-k indices ---
    sorted_top_indices = top_candidates[np.argsort(combined_sim_top)[::-1][:top_k]]

    return sorted_top_indices


# --- Minimal preprocessing for embeddings ---
def preprocess_text_for_embeddings(text: str) -> str:
    """
    Minimal preprocessing for embedding-based search:
    - Lowercase
    - Remove punctuation
    - Preserve stopwords and negations
    """
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', ' ', text)  # remove punctuation
    text = re.sub(r'\s+', ' ', text).strip()  # collapse multiple spaces
    return text

# --- Minimal preprocessing for keywords ---
def preprocess_text_for_keywords(text: str) -> str:
    """
    Minimal preprocessing for keyword extraction:
    - Lowercase
    - Remove punctuation
    - Keep meaningful words
    """
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text
