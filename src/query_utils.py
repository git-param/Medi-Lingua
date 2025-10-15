import re
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from keybert import KeyBERT

# --- Download NLTK resources if needed ---
try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
    nltk.download('wordnet', quiet=True)

# --- Initialize tools ---
tokenizer = RegexpTokenizer(r'\w+')
lemmatizer = WordNetLemmatizer()
custom_stopwords = set(stopwords.words('english')) - {'no', 'not', 'without', 'due', 'to', 'with', 'on', 'in'}

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

def expand_medical_terms(text: str) -> str:
    """Expands known medical terms with their synonyms for better recall."""
    for key, syns in medical_synonyms.items():
        for syn in syns:
            text = re.sub(rf"\b{key}\b", f"{key} {syn}", text, flags=re.IGNORECASE)
    return text

def preprocess_text(text: str) -> str:
    """Minimal preprocessing: lowercase, remove punctuation, collapse spaces."""
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

class QueryEnhancer:
    """
    Wrapper class to handle query enhancement with local SapBERT + KeyBERT.
    """
    def __init__(self, sentence_transformer_model):
        """
        sentence_transformer_model: the already-loaded local SapBERT SentenceTransformer
        """
        self.kw_model = KeyBERT(sentence_transformer_model)

    def extract_keywords(self, text: str, top_n: int = 5) -> list:
        """Extracts top keywords using KeyBERT."""
        if not self.kw_model:
            return []
        try:
            keywords = self.kw_model.extract_keywords(
                text,
                keyphrase_ngram_range=(1, 2),
                stop_words='english',
                top_n=top_n
            )
            return [kw[0] for kw in keywords]
        except Exception:
            return []

    def enhance_query(self, user_query: str) -> str:
        """
        Full query enhancement pipeline:
        - Preprocess text
        - Expand medical synonyms
        - Extract keywords
        - Return combined enhanced query string
        """
        preprocessed = preprocess_text(user_query)
        expanded = expand_medical_terms(preprocessed)
        keywords = self.extract_keywords(user_query)
        enhanced_query = f"{expanded} {' '.join(keywords)}".strip()
        return enhanced_query
