import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import config
from models import nlp, kw_model, tfidf_vectorizer, tfidf_feature_matrix
from data_processing import movies_tfidf_df
import logging
import re
from fuzzywuzzy import fuzz  # For fuzzy matching
import pandas as pd

logger = logging.getLogger(__name__)


# --- Helper Functions ---


def entity_matches_title(entity, title):
    """Check if an entity matches a movie title using fuzzy matching"""
    # Remove year in parentheses from the title for better matching
    clean_title = re.sub(r"\s*\(\d{4}\)\s*$", "", title).lower()
    entity = entity.lower()

    # Exact match
    if entity in clean_title or clean_title in entity:
        return True

    # Fuzzy match for longer entity names (to avoid false positives with short names)
    if len(entity) > 3:
        ratio = fuzz.partial_ratio(entity, clean_title)
        return ratio > 80  # Threshold for considering it a match

    return False


def entity_in_tags(entity, tags):
    """Check if an entity is present in the movie's tags"""
    entity = entity.lower()

    # Direct match
    if any(entity in tag.lower() for tag in tags):
        return True

    # Fuzzy match for longer entity names
    if len(entity) > 3:
        for tag in tags:
            if fuzz.partial_ratio(entity, tag.lower()) > 80:
                return True

    return False


def parse_decade_reference(prompt: str):
    """
    Parse decade references from the prompt like "90s", "2000s", etc.
    Returns a tuple of (start_year, end_year) if found, otherwise None.
    """
    # Pattern for "90s", "80s", etc.
    short_pattern = r"\b([1-9]0)s\b"
    # Pattern for "1990s", "2000s", etc.
    full_pattern = r"\b(1[0-9]{3}|20[0-9]{2})s\b"

    # First check for full patterns like "1990s" or "2000s"
    full_match = re.search(full_pattern, prompt.lower())
    if full_match:
        base_year = int(full_match.group(1))
        return (base_year, base_year + 9)

    # Then check for short patterns like "90s"
    short_match = re.search(short_pattern, prompt.lower())
    if short_match:
        decade = int(short_match.group(1))
        # Convert "90" to "1990", "00" to "2000", etc.
        if decade < 30:  # Assume 00s, 10s, 20s refer to 2000s, 2010s, 2020s
            base_year = 2000 + decade
        else:  # Assume 30s through 90s refer to 1900s
            base_year = 1900 + decade
        return (base_year, base_year + 9)

    return None


def extract_year_reference(entities, entity_types):
    """
    Extract specific year references from SpaCy's entity recognition.
    Returns a tuple of (start_year, end_year) for filtering.
    """
    years = []

    # Look for DATE entities that might be years
    for entity, entity_type in zip(entities, entity_types):
        if entity_type == "DATE":
            # Try to extract a 4-digit year from the entity
            year_match = re.search(r"\b(19[0-9]{2}|20[0-9]{2})\b", entity)
            if year_match:
                year = int(year_match.group(1))
                # Only consider reasonable movie years (1900-2025)
                if 1900 <= year <= 2025:
                    years.append(year)

    if years:
        # If we found one specific year, use it for exact year matching
        if len(years) == 1:
            year = years[0]
            return (year, year)  # Same start and end year for exact match

    return None


def extract_features(prompt: str) -> dict:
    """Hybrid feature extraction with pretrained models and rules"""
    doc = nlp(prompt.lower())

    # Rule-based genre detection using direct keyword-to-genre mapping
    detected_genres = []
    prompt_lower = prompt.lower()
    prompt_words = prompt_lower.split()

    # Direct lookup for single word keywords
    for word in prompt_words:
        if word in config.KEYWORD_TO_GENRE:
            detected_genres.append(config.KEYWORD_TO_GENRE[word])

    # Check for multi-word keywords
    for keyword in config.KEYWORD_TO_GENRE:
        if " " in keyword and keyword in prompt_lower:
            detected_genres.append(config.KEYWORD_TO_GENRE[keyword])

    # Also detect the genre name directly if mentioned
    for genre in config.GENRE_MAPPING:
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

    # Enhanced SpaCy entity recognition - expanded to include more entity types
    entity_dict = {}
    for ent in doc.ents:
        # Include more entity types, especially DATE
        if ent.label_ in [
            "PERSON",
            "ORG",
            "WORK_OF_ART",
            "EVENT",
            "PRODUCT",
            "LOC",
            "FAC",
            "DATE",
            "GPE",
            "NORP",
            "LANGUAGE",
            "TITLE",
        ]:
            entity_dict[ent.text] = ent.label_

    entities = list(entity_dict.keys())
    entity_types = list(entity_dict.values())

    combined_features = list(set(detected_genres + key_phrases + entities))
    processed_text = " ".join(combined_features)

    return {
        "genres": detected_genres,
        "keywords": key_phrases,
        "entities": entities,
        "entity_types": entity_types,
        "processed_text": processed_text,
    }


def find_entity_matches(features, top_n=50):
    """Find movies that match extracted entities in titles or tags"""
    entity_matches = []

    # Only process if we have entities
    if not features["entities"]:
        return []

    # Check entities against movie titles and tags
    for idx, movie in movies_tfidf_df.head(top_n).iterrows():
        title = movie["title"]
        tags = movie.get("tags", [])
        match_score = 0
        matched_entities = []

        for entity in features["entities"]:
            # Check if entity matches title
            if entity_matches_title(entity, title):
                match_score += 3  # Higher weight for title matches
                matched_entities.append(entity)

            # Check if entity matches any tags
            elif entity_in_tags(entity, tags):
                match_score += 1
                matched_entities.append(entity)

        if match_score > 0:
            entity_matches.append(
                {
                    "movieId": movie["movieId"],
                    "title": title,
                    "match_score": match_score,
                    "matched_entities": matched_entities,
                }
            )

    # Sort by match score
    entity_matches.sort(key=lambda x: x["match_score"], reverse=True)
    return entity_matches


# --- Recommendation Logic ---


def get_enhanced_recommendations(prompt: str, top_n=10):
    """Generates recommendations with enhanced entity matching"""
    logger.info(f"--- Inside Enhanced Recommender ---")
    logger.info(f"Received prompt: {prompt}, top_n: {top_n}")
    features = extract_features(prompt)
    logger.info(f"Extracted Features: {features}")

    # Check for decade references first
    decade_range = parse_decade_reference(prompt)
    year_range = None

    if decade_range:
        start_year, end_year = decade_range
        logger.info(
            f"Detected decade reference: {start_year}s ({start_year}-{end_year})"
        )
    else:
        # If no decade reference found, check for specific year
        year_range = extract_year_reference(
            features["entities"], features["entity_types"]
        )
        if year_range:
            logger.info(f"Detected specific year reference: {year_range[0]}")

    # Use either decade range or specific year range for filtering
    date_filter_range = decade_range or year_range

    # Get standard TF-IDF recommendations (date filtering already applied)
    tfidf_recommendations = get_keyword_recommendations(prompt, top_n=top_n)

    # Get entity-based matches
    entity_matches = find_entity_matches(features, top_n=100)
    logger.info(f"Found {len(entity_matches)} entity matches")

    # Filter entity matches by date range if specified
    if date_filter_range:
        start_year, end_year = date_filter_range
        filtered_entity_matches = []

        for match in entity_matches:
            # Extract year from title using regex
            year_match = re.search(r"\((\d{4})\)", match["title"])
            if year_match:
                year = int(year_match.group(1))
                if start_year <= year <= end_year:
                    filtered_entity_matches.append(match)

        entity_matches = filtered_entity_matches
        logger.info(
            f"After date filtering, {len(entity_matches)} entity matches remain"
        )

    # If we have entity matches, boost those recommendations
    if entity_matches:
        # Create a dictionary of movieId -> entity match score
        entity_match_dict = {
            match["movieId"]: match["match_score"] for match in entity_matches
        }

        # Get IDs from the TF-IDF recommendations
        tfidf_recommendation_ids = set(tfidf_recommendations["movieId"].values)

        # Find any high-scoring entity matches not already in TF-IDF recommendations
        missing_entity_matches = [
            match
            for match in entity_matches
            if match["movieId"] not in tfidf_recommendation_ids
            and match["match_score"] >= 3
        ]

        # If we have missing high-scoring matches, add them
        if missing_entity_matches:
            top_missing = missing_entity_matches[: min(3, len(missing_entity_matches))]

            # Get these movies from the DataFrame
            for match in top_missing:
                movie_data = movies_tfidf_df[
                    movies_tfidf_df["movieId"] == match["movieId"]
                ].iloc[0]
                match_row = {
                    "movieId": movie_data["movieId"],
                    "title": movie_data["title"],
                    "genres": movie_data["genres"].split("|"),
                    "bayesian_avg": movie_data["bayesian_avg"],
                    "similarity_score": 0.5
                    + (match["match_score"] / 10.0),  # Base score + bonus
                }
                # Add to recommendations
                tfidf_recommendations = tfidf_recommendations.append(
                    match_row, ignore_index=True
                )

            # Re-sort by similarity score
            tfidf_recommendations = tfidf_recommendations.sort_values(
                by="similarity_score", ascending=False
            ).head(top_n)

    logger.info(f"Returning {len(tfidf_recommendations)} recommendations.")
    return tfidf_recommendations


def get_keyword_recommendations(prompt: str, top_n=10):
    """Generates recommendations based on keyword/TF-IDF matching."""
    logger.info(f"--- Inside Keyword Recommender ---")
    logger.info(f"Received prompt: {prompt}, top_n: {top_n}")
    features = extract_features(prompt)
    logger.info(f"Extracted Features: {features}")

    # Check for decade references first
    decade_range = parse_decade_reference(prompt)
    year_range = None

    if decade_range:
        start_year, end_year = decade_range
        logger.info(
            f"Detected decade reference: {start_year}s ({start_year}-{end_year})"
        )
    else:
        # If no decade reference found, check for specific year
        year_range = extract_year_reference(
            features["entities"], features["entity_types"]
        )
        if year_range:
            logger.info(f"Detected specific year reference: {year_range[0]}")

    # Use either decade range or specific year range for filtering
    date_filter_range = decade_range or year_range

    if not features["processed_text"]:
        logger.warning(
            "No relevant features extracted. Returning top rated movies based on Bayesian Average."
        )
        # Return top N based on precalculated Bayesian average
        top_rated = movies_tfidf_df.nlargest(top_n, "bayesian_avg")[
            ["movieId", "title", "genres", "bayesian_avg"]
        ].copy()

        # Apply date filter if specified
        if date_filter_range:
            # Extract year from title (assuming format includes year in parentheses)
            top_rated["year"] = (
                top_rated["title"].str.extract(r"\((\d{4})\)").astype("float")
            )
            top_rated = top_rated[
                (top_rated["year"] >= date_filter_range[0])
                & (top_rated["year"] <= date_filter_range[1])
            ]

            # If filtering resulted in too few movies, get more from the original set
            if len(top_rated) < top_n:
                remaining = top_n - len(top_rated)
                more_movies = movies_tfidf_df.nlargest(top_n * 3, "bayesian_avg")[
                    ["movieId", "title", "genres", "bayesian_avg"]
                ].copy()
                more_movies["year"] = (
                    more_movies["title"].str.extract(r"\((\d{4})\)").astype("float")
                )
                more_movies = more_movies[
                    (more_movies["year"] >= date_filter_range[0])
                    & (more_movies["year"] <= date_filter_range[1])
                ]
                more_movies = more_movies[
                    ~more_movies["movieId"].isin(top_rated["movieId"])
                ]
                top_rated = pd.concat([top_rated, more_movies.head(remaining)])

        top_rated["genres"] = top_rated["genres"].apply(lambda x: x.split("|"))
        return top_rated.assign(
            similarity_score=0.0
        )  # Assign 0 similarity for this case

    prompt_vector = tfidf_vectorizer.transform([features["processed_text"]])
    similarities = cosine_similarity(prompt_vector, tfidf_feature_matrix).flatten()

    # Get top matches based purely on similarity (get more than needed for filtering)
    if date_filter_range:
        top_n_multiplier = 5  # Get 5x more results to allow for filtering
        similar_indices = np.argsort(similarities)[::-1][: top_n * top_n_multiplier]
    else:
        similar_indices = np.argsort(similarities)[::-1][:top_n]

    # Get the corresponding movie data
    top_recommendations = movies_tfidf_df.iloc[similar_indices].copy()
    top_recommendations["similarity_score"] = similarities[similar_indices]

    # Apply date filter if specified
    if date_filter_range:
        start_year, end_year = date_filter_range
        # Extract year from title (assuming format includes year in parentheses)
        top_recommendations["year"] = (
            top_recommendations["title"].str.extract(r"\((\d{4})\)").astype("float")
        )
        # Filter by year range
        top_recommendations = top_recommendations[
            (top_recommendations["year"] >= start_year)
            & (top_recommendations["year"] <= end_year)
        ]
        # Keep only top_n after filtering
        top_recommendations = top_recommendations.head(top_n)

    # Convert genres string to list
    top_recommendations["genres"] = top_recommendations["genres"].apply(
        lambda x: x.split("|")
    )

    logger.info(
        f"Returning {len(top_recommendations)} recommendations based on similarity."
    )

    # Return columns matching the original snippet's output structure
    return top_recommendations[
        ["movieId", "title", "genres", "bayesian_avg", "similarity_score"]
    ]
