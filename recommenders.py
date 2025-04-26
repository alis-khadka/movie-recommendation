import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import config
from models import nlp, kw_model, tfidf_vectorizer, tfidf_feature_matrix
from data_processing import movies_tfidf_df
import logging

logger = logging.getLogger(__name__)


# --- Helper Functions ---


def extract_features(prompt: str) -> dict:
    """Hybrid feature extraction with pretrained models and rules"""
    doc = nlp(prompt.lower())

    # Rule-based genre detection
    detected_genres = []
    prompt_lower = prompt.lower()
    for genre, keywords in config.GENRE_MAPPING.items():
        if any(keyword in prompt_lower for keyword in keywords):
            detected_genres.append(genre)
        if genre in prompt_lower.split():
            detected_genres.append(genre)
    detected_genres = list(set(detected_genres))

    # KeyBERT keyword extraction
    keywords = kw_model.extract_keywords(
        prompt,
        keyphrase_ngram_range=(1, 2),
        stop_words="english",
        top_n=5,
        use_mmr=True,
        diversity=0.7,
    )
    key_phrases = [kw[0] for kw in keywords if kw[1] > 0.2]

    # SpaCy entity recognition
    entities = [
        ent.text
        for ent in doc.ents
        if ent.label_
        in ["PERSON", "ORG", "WORK_OF_ART", "EVENT", "PRODUCT", "LOC", "FAC"]
    ]

    combined_features = list(set(detected_genres + key_phrases + entities))
    processed_text = " ".join(combined_features)

    return {
        "genres": detected_genres,
        "keywords": key_phrases,
        "entities": entities,
        "processed_text": processed_text,
    }


# --- Recommendation Logic ---


def get_keyword_recommendations(prompt: str, top_n=10):
    """Generates recommendations based on keyword/TF-IDF matching."""
    logger.info(f"--- Inside Keyword Recommender ---")
    logger.info(f"Received prompt: {prompt}, top_n: {top_n}")
    features = extract_features(prompt)
    logger.info(f"Extracted Features: {features}")

    if not features["processed_text"]:
        logger.warning(
            "No relevant features extracted. Returning top rated movies based on Bayesian Average."
        )
        # Return top N based on precalculated Bayesian average
        top_rated = movies_tfidf_df.nlargest(top_n, "bayesian_avg")[
            ["title", "genres", "bayesian_avg"]
        ].copy()
        top_rated["genres"] = top_rated["genres"].apply(lambda x: x.split("|"))
        return top_rated.assign(
            similarity_score=0.0
        )  # Assign 0 similarity for this case

    prompt_vector = tfidf_vectorizer.transform([features["processed_text"]])
    similarities = cosine_similarity(prompt_vector, tfidf_feature_matrix).flatten()

    # Get top matches based purely on similarity (like original snippet)
    similar_indices = np.argsort(similarities)[::-1][:top_n]

    # Get the corresponding movie data
    top_recommendations = movies_tfidf_df.iloc[similar_indices].copy()
    top_recommendations["similarity_score"] = similarities[similar_indices]
    # Convert genres string to list
    top_recommendations["genres"] = top_recommendations["genres"].apply(
        lambda x: x.split("|")
    )

    logger.info(
        f"Returning {len(top_recommendations)} recommendations based on similarity."
    )

    # Return columns matching the original snippet's output structure
    return top_recommendations[["title", "genres", "bayesian_avg", "similarity_score"]]
