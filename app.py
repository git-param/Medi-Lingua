import streamlit as st
from src.utils import load_model, load_data
from src.query_utils import QueryEnhancer
from src.search import hybrid_search, set_description_texts, set_patient_texts, strong_recall_indices
from src.ui import apply_custom_css, render_header, render_sidebar, render_chat_history, bot_typing_animation
from src.pdf_utils import build_chat_pdf
from src.llm import summarize_with_gemini
import numpy as np

def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def filter_similar_answers(indices, doctor_embeddings, threshold=0.88):
    """
    Filters out semantically similar answers based on cosine similarity.

    Args:
        indices (list[int]): candidate indices of top answers (assumed sorted by relevance)
        doctor_embeddings (np.array): embeddings of all doctor's answers (num_samples x dim)
        threshold (float): similarity above which answers are considered duplicates

    Returns:
        filtered_indices (list[int]): indices of diverse answers
    """
    if len(indices) == 0:
        return []

    filtered = [indices[0]]  # Always keep the first (most relevant) one
    for idx in indices[1:]:
        emb = doctor_embeddings[idx]
        keep = True
        for f_idx in filtered:
            existing_emb = doctor_embeddings[f_idx]
            sim = np.dot(emb, existing_emb)  # Cosine sim (since normalized)
            if sim >= threshold:
                keep = False
                break
        if keep:
            filtered.append(idx)
    return filtered

# --- 1. Setup UI ---
apply_custom_css()
render_header()
render_sidebar()

# --- 2. Load Model & Data ---
model = load_model()
data = load_data()
google_api_key = st.secrets.get("GOOGLE_API_KEY")

if not data:
    st.error("❌ Could not load dataset or embeddings. Please check your paths.")
    st.stop()

query_enhancer = QueryEnhancer(model)

# --- 3. Set texts for recall ---
set_description_texts(data['description_column'])
set_patient_texts(data['patient_column'])

# --- 4. Initialize Chat History ---
if "messages" not in st.session_state:
    st.session_state.messages = []

render_chat_history(st.session_state.messages)

# --- 5. PDF export sidebar ---
with st.sidebar:
    st.markdown("---")
    st.subheader("Export")
    if st.session_state.get("messages"):
        pdf_buffer = build_chat_pdf(
            st.session_state.messages, title="MediLingua Chat Transcript"
        )
        if pdf_buffer:
            st.download_button(
                label="📄 Download chat as PDF",
                data=pdf_buffer,
                file_name="medilingua_chat.pdf",
                mime="application/pdf",
            )
        else:
            st.caption(
                "Install `reportlab` to enable PDF export: pip install reportlab"
            )

# --- 6. Handle User Input ---
if prompt := st.chat_input("What is your medical question?"):
    # Show user input immediately
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Placeholder for bot response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()

        # Step 1: Enhance query
        enhanced_query = query_enhancer.enhance_query(prompt)

        # Step 2: Strong recall + hybrid search
        with st.spinner("🔍 Searching for top relevant answers..."):
            # 1️⃣ Get top-k candidates
            indices = strong_recall_indices(prompt, top_k=10)
            if not indices or len(indices) < 3:
                indices = hybrid_search(
                    model,
                    data['question_embeddings'],
                    user_query_raw=prompt,
                    user_query_enhanced=enhanced_query,
                    top_k=10,
                    weight_semantic=0.7,
                    faiss_top_candidates=256,
                    use_exact_match=True,
                    use_fuzzy_match=True
                )

            # 2️⃣ Filter out semantically similar doctor answers to ensure diversity
            indices = filter_similar_answers(indices, data['doctor_embeddings'], threshold=0.88)

        # Step 3: Summarization / Gemini
        if indices is not None and len(indices) > 0:
            # Gather ALL filtered diverse answers for summarization (used as reference)
            top_answers = [data['original_answers'][i] for i in indices]
            combined_text = " ".join(top_answers)

            summary = summarize_with_gemini(google_api_key, combined_text, prompt)

            # Show only the AI's summarized answer (no doctor's notes displayed)
            bot_typing_animation(message_placeholder, summary)

            response = summary
        else:
            response = "⚕️ I couldn’t find any contextually similar answer in the dataset."
            bot_typing_animation(message_placeholder, response)

    # Save conversation
    st.session_state.messages.append({"role": "assistant", "content": response})