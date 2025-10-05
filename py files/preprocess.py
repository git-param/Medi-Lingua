import pandas as pd
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import spacy
from collections import Counter
import matplotlib.pyplot as plt
import re

def preprocess_data(file_path):
    """
    Loads, cleans, and preprocesses the medical chatbot dataset.

    Args:
        file_path (str): The path to the dataset CSV file.

    Returns:
        pandas.DataFrame: The preprocessed DataFrame.
    """
    print("Loading dataset...")
    try:
        # Using a path relative to the project root (Medilingua)
        data = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: The file at {file_path} was not found.")
        return None

    print("Initial dataset shape:", data.shape)

    # --- Basic Cleaning ---
    print("Cleaning data...")
    data.dropna(inplace=True)
    data.drop_duplicates(inplace=True)
    print("Shape after dropping nulls and duplicates:", data.shape)

    # Remove 'Q.' and 'A.' prefixes
    data['Description'] = data['Description'].str.replace(r'^Q\.\s*', '', regex=True)
    # The Patient and Doctor columns don't seem to have prefixes in the sample, but adding just in case
    data['Patient'] = data['Patient'].str.replace(r'^[A-Za-z]+\s*doctor,', '', regex=True)


    # --- Text Preprocessing ---
    print("Preprocessing text data...")

    # 1. Lowercasing
    for col in ['Description', 'Patient', 'Doctor']:
        data[col] = data[col].str.lower()

    # 2. Tokenization
    tokenizer = RegexpTokenizer(r'\w+')
    for col in ['Description', 'Patient', 'Doctor']:
        data[col] = data[col].apply(tokenizer.tokenize)

    # 3. Stopword Removal
    stop_words = set(stopwords.words('english'))
    custom_stopwords = {"doctor", "please", "hello", "hi", "im", "ive", "thanks", "thank", "nan"}
    stop_words = stop_words.union(custom_stopwords)

    for col in ['Description', 'Patient', 'Doctor']:
        data[col] = data[col].apply(lambda tokens: [word for word in tokens if word not in stop_words])

    # 4. POS Tagging
    # Note: NLTK downloads can be slow the first time.
    print("Performing POS tagging...")
    nltk.download('averaged_perceptron_tagger', quiet=True)
    data['Description_pos'] = data['Description'].apply(nltk.pos_tag)
    data['Patient_pos'] = data['Patient'].apply(nltk.pos_tag)
    data['Doctor_pos'] = data['Doctor'].apply(nltk.pos_tag)


    # 5. Lemmatization
    print("Lemmatizing text...")
    nltk.download('wordnet', quiet=True)
    lemmatizer = WordNetLemmatizer()

    def get_wordnet_pos(tag):
        if tag.startswith('J'):
            return wordnet.ADJ
        elif tag.startswith('V'):
            return wordnet.VERB
        elif tag.startswith('N'):
            return wordnet.NOUN
        elif tag.startswith('R'):
            return wordnet.ADV
        else:
            return wordnet.NOUN  # Default to noun

    def lemmatize_pos(tagged_tokens):
        return [lemmatizer.lemmatize(word, get_wordnet_pos(pos)) for word, pos in tagged_tokens]

    data['Description'] = data['Description_pos'].apply(lemmatize_pos)
    data['Patient'] = data['Patient_pos'].apply(lemmatize_pos)
    data['Doctor'] = data['Doctor_pos'].apply(lemmatize_pos)

    # Drop the intermediate POS tag columns
    data.drop(columns=['Description_pos', 'Patient_pos', 'Doctor_pos'], inplace=True)

    # Join tokens back into strings for saving
    for col in ['Description', 'Patient', 'Doctor']:
        data[col] = data[col].apply(lambda tokens: ' '.join(tokens))
        
    print("Preprocessing complete.")
    return data

if __name__ == '__main__':
    # This block will run when the script is executed directly
    # Corrected paths to be relative to the project root
    input_path = 'dataset/dataset.csv'
    output_path = 'dataset/preprocessed_dataset.csv'
    
    preprocessed_df = preprocess_data(input_path)

    if preprocessed_df is not None:
        print(f"Saving preprocessed data to {output_path}...")
        preprocessed_df.to_csv(output_path, index=False)
        print("Done.")
        print("\nSample of preprocessed data:")
        print(preprocessed_df.head())