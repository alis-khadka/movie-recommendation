import spacy
from spacy.cli import download as spacy_download
from sentence_transformers import SentenceTransformer
from keybert import KeyBERT
from sklearn.feature_extraction.text import TfidfVectorizer
from google import genai
import config
from data_processing import movies_tfidf_df

print("Initializing models...")

gemini_client = genai.Client(api_key=config.GEMINI_API_KEY)
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# TF-IDF/Keyword Recommender Models
try:
    nlp = spacy.load(config.SPACY_MODEL_NAME)
except OSError:
    print(f"Spacy model '{config.SPACY_MODEL_NAME}' not found. Downloading...")
    spacy_download(config.SPACY_MODEL_NAME)
    nlp = spacy.load(config.SPACY_MODEL_NAME)

kw_model = KeyBERT(model=nlp)
tfidf_vectorizer = TfidfVectorizer(
    stop_words="english", ngram_range=(1, 2), max_features=5000
)

# Fit TF-IDF on movie genres
tfidf_feature_matrix = tfidf_vectorizer.fit_transform(
    movies_tfidf_df["processed_genres"]
)

print("Models initialized.")
