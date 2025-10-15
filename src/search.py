import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import faiss
import re

# --- Global caches ---
tfidf_vectorizer = None
tfidf_matrix = None
corpus_texts = None
faiss_index = None
embeddings_array = None  # FAISS requires float32
description_texts = None  # For exact/fuzzy match
patient_texts = None      # For exact/fuzzy match
description_norm_texts = None  # Normalized (punctuation stripped)
patient_norm_texts = None      # Normalized (punctuation stripped)


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

    # Do not renormalize
    dimension = embeddings_array.shape[1]
    faiss_index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
    faiss_index.add(embeddings_array)



def set_description_texts(texts):
    """
    Provide the Description column for exact/fuzzy match search.
    """
    global description_texts, description_norm_texts
    description_texts = [str(t).lower() for t in texts]
    description_norm_texts = [preprocess_text_for_embeddings(t) for t in texts]


def set_patient_texts(texts):
    """
    Provide the Patient column for exact/fuzzy match search.
    """
    global patient_texts, patient_norm_texts
    patient_texts = [str(t).lower() for t in texts]
    patient_norm_texts = [preprocess_text_for_embeddings(t) for t in texts]


def strong_recall_indices(user_query_raw: str, top_k: int = 10):
    """
    Scan the entire dataset (Description + Patient) for:
    1) Exact equality on normalized text
    2) Exact substring presence
    3) High-threshold fuzzy match (if rapidfuzz available)

    Returns a list of indices (unique, in priority order) up to top_k.
    """
    global description_texts, patient_texts, description_norm_texts, patient_norm_texts

    if not user_query_raw:
        return []

    q_lower = str(user_query_raw).lower()
    q_norm = preprocess_text_for_embeddings(user_query_raw)

    N_desc = len(description_texts) if description_texts is not None else 0
    N_pat = len(patient_texts) if patient_texts is not None else 0
    N = max(N_desc, N_pat)
    if N == 0:
        return []

    exact_equal = []
    exact_sub = []
    fuzzy_hits = []

    # 1) Exact equality on normalized text
    if description_norm_texts is not None:
        exact_equal += [i for i in range(len(description_norm_texts)) if description_norm_texts[i] == q_norm]
    if patient_norm_texts is not None:
        exact_equal += [i for i in range(len(patient_norm_texts)) if patient_norm_texts[i] == q_norm]

    # Deduplicate preserving order
    seen = set()
    ordered = []
    for i in exact_equal:
        if i not in seen:
            seen.add(i)
            ordered.append(i)
    if len(ordered) >= top_k:
        return ordered[:top_k]

    # 2) Exact substring presence (lowercased)
    if description_texts is not None:
        exact_sub += [i for i in range(len(description_texts)) if q_lower in description_texts[i]]
    if patient_texts is not None:
        exact_sub += [i for i in range(len(patient_texts)) if q_lower in patient_texts[i]]

    for i in exact_sub:
        if i not in seen:
            seen.add(i)
            ordered.append(i)
    if len(ordered) >= top_k:
        return ordered[:top_k]

    # 3) High-threshold fuzzy matches
    try:
        from rapidfuzz import fuzz
        # Use partial_ratio and token_set_ratio; take max as score
        scored = []
        for i in range(N):
            s_desc = description_texts[i] if (description_texts is not None and i < len(description_texts)) else ""
            s_pat = patient_texts[i] if (patient_texts is not None and i < len(patient_texts)) else ""
            score_desc = max(fuzz.partial_ratio(q_lower, s_desc), fuzz.token_set_ratio(q_lower, s_desc)) if s_desc else 0
            score_pat = max(fuzz.partial_ratio(q_lower, s_pat), fuzz.token_set_ratio(q_lower, s_pat)) if s_pat else 0
            score = max(score_desc, score_pat)
            if score >= 90:
                scored.append((i, score))
        # sort by score desc
        scored.sort(key=lambda x: x[1], reverse=True)
        for i, _ in scored:
            if i not in seen:
                seen.add(i)
                ordered.append(i)
            if len(ordered) >= top_k:
                break
    except Exception:
        pass

    return ordered[:top_k]


def hybrid_search(
    model,
    embeddings,
    user_query_raw,
    user_query_enhanced,
    top_k=5,
    weight_semantic=0.7,
    faiss_top_candidates=256,
    use_exact_match=True,
    use_fuzzy_match=True
):
    """
    Hybrid search combining:
    1. FAISS semantic similarity
    2. TF-IDF boosting
    3. Optional exact substring match in Description

    Returns: list of top indices in dataset
    """
    global tfidf_vectorizer, tfidf_matrix, corpus_texts, faiss_index, embeddings_array, description_texts

    if model is None or embeddings is None or len(embeddings) == 0:
        return []

    # Encode enhanced query for semantic/TF-IDF stages
    question_embedding = encode_question(model, user_query_enhanced)
    if question_embedding is None:
        return []

    # --- 1. FAISS semantic search ---
    if faiss_index is not None:
        D, I = faiss_index.search(np.array([question_embedding]), k=min(faiss_top_candidates, embeddings.shape[0]))
        top_candidates = I[0]
        semantic_sim_top = D[0]
    else:
        semantic_sim_full = np.dot(embeddings, question_embedding)
        top_candidates = np.argpartition(semantic_sim_full, -faiss_top_candidates)[-faiss_top_candidates:]
        top_candidates = top_candidates[np.argsort(semantic_sim_full[top_candidates])[::-1]]
        semantic_sim_top = semantic_sim_full[top_candidates]

    # --- 2. TF-IDF similarity ---
    if tfidf_vectorizer is not None and tfidf_matrix is not None:
        tfidf_vec = tfidf_vectorizer.transform([user_query_enhanced])
        tfidf_sim_top = (tfidf_matrix[top_candidates] @ tfidf_vec.T).toarray().ravel()
    else:
        tfidf_sim_top = np.zeros(len(top_candidates))

    # --- 3. Optional exact + fuzzy match across Description & Patient ---
    combined_sim_top = weight_semantic * semantic_sim_top + (1 - weight_semantic) * tfidf_sim_top

    if use_exact_match or use_fuzzy_match:
        query_lower = user_query_raw.lower()

        # Exact substring presence boosts
        exact_desc = np.zeros(len(top_candidates))
        exact_pat = np.zeros(len(top_candidates))
        if description_texts is not None:
            exact_desc = np.array([1.0 if query_lower in description_texts[i] else 0.0 for i in top_candidates])
        if patient_texts is not None:
            exact_pat = np.array([1.0 if query_lower in patient_texts[i] else 0.0 for i in top_candidates])

        # Fuzzy partial ratio via rapidfuzz (graceful fallback)
        fuzzy_desc = np.zeros(len(top_candidates))
        fuzzy_pat = np.zeros(len(top_candidates))
        if use_fuzzy_match:
            try:
                from rapidfuzz import fuzz
                if description_texts is not None:
                    fuzzy_desc = np.array([
                        fuzz.partial_ratio(query_lower, description_texts[i]) / 100.0 for i in top_candidates
                    ])
                if patient_texts is not None:
                    fuzzy_pat = np.array([
                        fuzz.partial_ratio(query_lower, patient_texts[i]) / 100.0 for i in top_candidates
                    ])
            except Exception:
                pass

        # Token overlap (Jaccard) as an additional weak signal
        def jaccard(a: str, b: str) -> float:
            sa = set(a.split())
            sb = set(b.split())
            if not sa or not sb:
                return 0.0
            inter = len(sa & sb)
            union = len(sa | sb)
            return inter / union if union else 0.0

        token_desc = np.zeros(len(top_candidates))
        token_pat = np.zeros(len(top_candidates))
        if description_texts is not None:
            token_desc = np.array([jaccard(query_lower, description_texts[i]) for i in top_candidates])
        if patient_texts is not None:
            token_pat = np.array([jaccard(query_lower, patient_texts[i]) for i in top_candidates])

        # Combine boosters with gentle weights; exact match is strongest
        booster = 0.20 * exact_desc + 0.20 * exact_pat + 0.10 * fuzzy_desc + 0.10 * fuzzy_pat + 0.05 * token_desc + 0.05 * token_pat
        combined_sim_top = combined_sim_top + booster

    # --- 4. Select final top-k indices ---
    sorted_top_indices = top_candidates[np.argsort(combined_sim_top)[::-1][:top_k]]

    return sorted_top_indices


# --- Minimal preprocessing for embeddings ---
def preprocess_text_for_embeddings(text: str) -> str:
    """Lowercase + remove punctuation for embeddings."""
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# --- Minimal preprocessing for keywords ---
def preprocess_text_for_keywords(text: str) -> str:
    """Lowercase + remove punctuation for keywords."""
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text
