import streamlit as st
import sys
import numpy as np
from sklearn.preprocessing import normalize
import re

# --- Graceful Import Handling ---
try:
    import nltk
    from nltk.tokenize import RegexpTokenizer
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    from nltk.corpus import wordnet
    from keybert import KeyBERT
    from src.utils import load_model, load_data
    from src.search import find_top_matches_indices, init_tfidf, preprocess_text_for_embeddings, preprocess_text_for_keywords
    from src.ui import apply_custom_css, render_header, render_sidebar, render_chat_history
    from src.llm import summarize_with_gemini
except ModuleNotFoundError as e:
    st.error(
        f"**Module Not Found: '{e.name}'**\n\nPlease activate your environment and run:\n\n"
        "`pip install sentence-transformers streamlit nltk pandas torch scikit-learn requests keybert`\n\n"
        "Then restart with: `python -m streamlit run app.py`"
    )
    st.stop()

# --- 1. SETUP AND LOAD DATA ---
apply_custom_css()
render_header()
render_sidebar()

model = load_model()
data = load_data()
google_api_key = st.secrets.get("GOOGLE_API_KEY")

# --- Normalize embeddings for consistent similarity ---
if 'embeddings' in data:
    data['embeddings'] = normalize(data['embeddings'], axis=1, norm='l2')

# --- Initialize TF-IDF for hybrid search ---
corpus_texts = [preprocess_text_for_embeddings(p + " " + d)
                for p, d in zip(data['patient_column'], data['description_column'])]
init_tfidf(corpus_texts)

# --- Preprocessing tools for keywords ---
try:
    stopwords.words('english')
except LookupError:
    with st.spinner("Downloading NLTK resources..."):
        nltk.download('stopwords', quiet=True)
        nltk.download('punkt', quiet=True)
        nltk.download('averaged_perceptron_tagger', quiet=True)
        nltk.download('wordnet', quiet=True)

tokenizer = RegexpTokenizer(r'\w+')
custom_stopwords = set(stopwords.words('english')) - {'no', 'not', 'without', 'due', 'to', 'with', 'on', 'in'}
lemmatizer = WordNetLemmatizer()
kw_model = KeyBERT('cambridgeltl/SapBERT-from-PubMedBERT-fulltext')

# --- Medical synonym expansion ---
medical_synonyms = {
    "flu": ["influenza"],
    "cold": ["common cold", "rhinitis"],
    "heart attack": ["myocardial infarction"],
    "diabetes": ["high blood sugar", "hyperglycemia"],
    "bp": ["blood pressure", "hypertension"],
    "hypertension": ["high blood pressure"],
    "asthma": ["respiratory disease"],
    "cough": ["dry cough", "wet cough"],
    "fever": ["temperature", "high fever"]
}

def expand_medical_terms(text):
    for key, syns in medical_synonyms.items():
        for syn in syns:
            text = re.sub(rf"\b{key}\b", key + " " + syn, text)
    return text

# --- Keyword extraction using KeyBERT ---
def extract_keywords(text, top_n=5):
    try:
        keywords = kw_model.extract_keywords(
            text,
            keyphrase_ngram_range=(1, 2),
            stop_words='english',
            top_n=top_n
        )
        return [kw[0] for kw in keywords]
    except Exception:
        return []

# --- 2. Initialize Chat History ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- 3. Render Existing Chat ---
render_chat_history(st.session_state.messages)

# --- 4. Handle User Input ---
if prompt := st.chat_input("What is your medical question?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()

        if not data:
            message_placeholder.markdown("⚠️ Data not loaded. Cannot process the request.")
        else:
            # Step 1: Minimal preprocessing for semantic search
            preprocessed_prompt = preprocess_text_for_embeddings(prompt)

            # Step 2: Medical term expansion
            expanded_prompt = expand_medical_terms(preprocessed_prompt)

            # Step 3: Keyword preprocessing for KeyBERT
            keywords_prompt = preprocess_text_for_keywords(prompt)
            keywords = extract_keywords(keywords_prompt)
            enhanced_query = expanded_prompt + " " + " ".join(keywords)

            # Step 4: Hybrid search for top matching entries
            with st.spinner("1/2 - Searching for top relevant answers..."):
                indices = find_top_matches_indices(model, data['embeddings'], enhanced_query, top_k=3)

            # Step 5: Summarize results using Gemini
            if indices is not None and len(indices) > 0:
                combined_text = " ".join([data['original_answers'][i] for i in indices])

                with st.spinner("2/2 - Summarizing response using Gemini..."):
                    summary = summarize_with_gemini(google_api_key, combined_text, prompt)

                message_placeholder.markdown(summary)

                # Display original doctor notes
                for i, index in enumerate(indices):
                    with st.expander(f"Show Original Doctor's Note {i+1}"):
                        st.markdown(f"> {data['original_answers'][index]}")

                response = summary
            else:
                response = "I couldn’t find a confident answer for your question in the dataset."
                message_placeholder.markdown(response)

    # Save conversation
    if 'response' in locals():
        st.session_state.messages.append({"role": "assistant", "content": response})
