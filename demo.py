import streamlit as st
import sys
import numpy as np
from sklearn.preprocessing import normalize

# --- Graceful Import Handling ---
try:
    import nltk
    from nltk.tokenize import RegexpTokenizer
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    from nltk.corpus import wordnet
    import re
    import spacy
    import scispacy
    from src.utils import load_model, load_data
    from src.search import find_top_matches_indices
    from src.ui import apply_custom_css, render_header, render_sidebar, render_chat_history
    from src.llm import summarize_with_gemini
except ModuleNotFoundError as e:
    st.error(
        f"**Module Not Found: '{e.name}'**\n\nPlease activate your environment and run:\n\n"
        "`pip install sentence-transformers streamlit nltk pandas torch scikit-learn requests spacy scispacy`\n"
        "and download the model:\n\n"
        "`pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.1/en_ner_bc5cdr_md-0.5.1.tar.gz`\n\n"
        "Then restart the app using: `python -m streamlit run app.py`"
    )
    st.stop()

# --- Load SciSpacy model for medical normalization ---
@st.cache_resource
def load_medical_nlp():
    try:
        return spacy.load("en_ner_bc5cdr_md")
    except OSError:
        st.warning("Downloading SciSpacy model...")
        from spacy.cli import download
        download("en_ner_bc5cdr_md")
        return spacy.load("en_ner_bc5cdr_md")

nlp_med = load_medical_nlp()

# --- 1. SETUP AND LOAD DATA ---
apply_custom_css()
render_header()
render_sidebar()

model = load_model()  # Should be SapBERT now
data = load_data()
google_api_key = st.secrets.get("GOOGLE_API_KEY")

# --- Normalize embeddings for consistent similarity ---
if 'embeddings' in data:
    data['embeddings'] = normalize(data['embeddings'], axis=1, norm='l2')

# --- Preprocessing tools ---
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

def get_wordnet_pos(tag):
    if tag.startswith('J'): return wordnet.ADJ
    elif tag.startswith('V'): return wordnet.VERB
    elif tag.startswith('N'): return wordnet.NOUN
    elif tag.startswith('R'): return wordnet.ADV
    else: return wordnet.NOUN

# --- Step 1: Basic text preprocessing ---
def basic_preprocess(text):
    text = str(text).lower()
    tokens = tokenizer.tokenize(text)
    tokens = [word for word in tokens if word not in custom_stopwords]
    tagged_tokens = nltk.pos_tag(tokens)
    lemmatized_tokens = [lemmatizer.lemmatize(word, get_wordnet_pos(pos)) for word, pos in tagged_tokens]
    return ' '.join(lemmatized_tokens)

# --- Step 2: Medical entity normalization (SciSpacy) ---
def normalize_medical_terms(query):
    doc = nlp_med(query)
    for ent in doc.ents:
        query = query.replace(ent.text, ent.label_)  # Replace entity with its type (e.g., "back pain" → "DISEASE")
    return query

# --- Step 3: Combined function for final preprocessing ---
def preprocess_text_for_embedding(text):
    cleaned = basic_preprocess(text)
    normalized = normalize_medical_terms(cleaned)
    return normalized

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
            preprocessed_prompt = preprocess_text_for_embedding(prompt)

            with st.spinner("1/2 - Searching for top relevant answers..."):
                indices = find_top_matches_indices(model, data['embeddings'], preprocessed_prompt, top_k=3)

            if indices is not None and len(indices) > 0:
                combined_text = " ".join([data['original_answers'][i] for i in indices])

                with st.spinner("2/2 - Summarizing response using Gemini..."):
                    summary = summarize_with_gemini(google_api_key, combined_text, prompt)

                message_placeholder.markdown(summary)

                for i, index in enumerate(indices):
                    with st.expander(f"Show Original Doctor's Note {i+1}"):
                        st.markdown(f"> {data['original_answers'][index]}")

                response = summary
            else:
                response = "I couldn’t find a confident answer for your question in the dataset."
                message_placeholder.markdown(response)

    if 'response' in locals():
        st.session_state.messages.append({"role": "assistant", "content": response})
