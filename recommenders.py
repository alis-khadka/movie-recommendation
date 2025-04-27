import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import config
from models import nlp, kw_model, tfidf_vectorizer, tfidf_feature_matrix
from data_processing import movies_tfidf_df
import logging
import re
from fuzzywuzzy import fuzz
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
        return ratio > 85  # Increased from 80 to 85 for more precise matching

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
            if fuzz.partial_ratio(entity, tag.lower()) > 85:  # Increased from 80 to 85
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

    # Direct lookup for single word keywords - give higher weight to genre matches
    for word in prompt_words:
        if word in config.KEYWORD_TO_GENRE:
            detected_genres.append(config.KEYWORD_TO_GENRE[word])
            # Add the genre twice to increase its weight in the processed text
            detected_genres.append(config.KEYWORD_TO_GENRE[word])

    # Check for multi-word keywords
    for keyword in config.KEYWORD_TO_GENRE:
        if " " in keyword and keyword in prompt_lower:
            detected_genres.append(config.KEYWORD_TO_GENRE[keyword])
            # Add the genre twice to increase its weight in the processed text
            detected_genres.append(config.KEYWORD_TO_GENRE[keyword])

    # Also detect the genre name directly if mentioned - with triple weight
    for genre in config.GENRE_MAPPING:
        if genre in prompt_lower.split():
            detected_genres.append(genre)
            detected_genres.append(genre)
            detected_genres.append(genre)

    detected_genres = list(set(detected_genres))

    # KeyBERT keyword extraction with improved diversity and threshold
    keywords = kw_model.extract_keywords(
        prompt,
        keyphrase_ngram_range=(1, 3),  # Increased from (1,2) to (1,3)
        stop_words="english",
        top_n=8,  # Increased from 5 to 8 to get more keywords
        use_mmr=True,
        diversity=0.5,  # Decreased from 0.7 to 0.5 for less diversity but more focus
    )
    key_phrases = [
        kw[0] for kw in keywords if kw[1] > 0.15
    ]  # Lower threshold from 0.2 to 0.15

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
            # Person and Work_of_Art entities are especially important for movies
            if ent.label_ in ["PERSON", "WORK_OF_ART"]:
                # Add them multiple times to increase weight
                entity_dict[ent.text + " " + ent.text] = ent.label_

    entities = list(entity_dict.keys())
    entity_types = list(entity_dict.values())

    # Give more weight to exact matches between entities and key phrases
    weighted_entities = []
    for entity in entities:
        weighted_entities.append(entity)
        # Add the entity again if it also appears in key phrases for more weight
        if entity in key_phrases:
            weighted_entities.append(entity)

    combined_features = list(set(detected_genres + key_phrases + weighted_entities))
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

    # Increase the number of movies considered to 3x the requested top_n for better recall
    search_pool = min(len(movies_tfidf_df), top_n * 3)

    # Check entities against movie titles and tags
    for idx, movie in movies_tfidf_df.head(search_pool).iterrows():
        title = movie["title"]
        tags = movie.get("tags", [])
        match_score = 0
        matched_entities = []

        # Extract the clean title without year for better matching
        clean_title = re.sub(r"\s*\(\d{4}\)\s*$", "", title).lower()

        for entity in features["entities"]:
            entity_lower = entity.lower()

            # Check if entity exactly matches full title (strongest signal)
            if entity_lower == clean_title:
                match_score += 8  # Highest score for exact title matches
                matched_entities.append(entity)
                continue

            # Check if entity matches title using our fuzzy match function
            if entity_matches_title(entity, title):
                # Give higher weight to person entities in titles
                if any(ent.label_ == "PERSON" for ent in nlp(entity).ents):
                    match_score += 7  # Increased weight for person entity matches
                else:
                    match_score += 5  # Standard title match
                matched_entities.append(entity)
                continue

            # Check if entity is a substantial part of the title
            # This helps with partial title matches like "Matrix" for "The Matrix"
            if len(entity_lower) > 3 and entity_lower in clean_title:
                # Make sure it's a significant portion (not just a minor word)
                if len(entity_lower) > len(clean_title) / 3:
                    match_score += 4  # Good score for substantial partial matches
                    matched_entities.append(entity)
                    continue

            # Check if entity matches any tags
            if entity_in_tags(entity, tags):
                match_score += 2
                matched_entities.append(entity)
                continue

            # Check for genre matches in entity if we have genres
            genres = movie.get("genres", "").lower().split("|")
            if any(genre in entity_lower for genre in genres if genre):
                match_score += 1  # Minor bonus for genre match
                matched_entities.append(entity)

        # Bonus score for movies with multiple matched entities
        if len(matched_entities) > 1:
            match_score += (
                len(matched_entities) - 1
            )  # Additional points for multiple matches

        # Only include if there's a meaningful match
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

    # Check for rating reference
    rating_threshold = None
    is_low_rating_query = (
        re.search(
            r"\b(?:low|poorly|worst|bad)\s+(?:rated|reviewed|acclaimed)\b",
            prompt.lower(),
        )
        is not None
    )

    if rating_threshold:
        logger.info(f"Detected rating threshold: {rating_threshold}")
    elif is_low_rating_query:
        logger.info("Detected request for low rated movies")

    # Get standard TF-IDF recommendations (date & rating filtering already applied)
    # Increase candidate pool for better diversity
    tfidf_recommendations = get_keyword_recommendations(prompt, top_n=top_n * 2)

    # Get entity-based matches with a larger pool
    entity_matches = find_entity_matches(features, top_n=200)
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

    # Filter entity matches by rating if needed
    if rating_threshold or is_low_rating_query:
        filtered_entity_matches = []

        for match in entity_matches:
            movie_data = movies_tfidf_df[movies_tfidf_df["movieId"] == match["movieId"]]

            if len(movie_data) > 0:
                bayesian_avg = movie_data["bayesian_avg"].values[0]

                if rating_threshold and bayesian_avg >= rating_threshold:
                    filtered_entity_matches.append(match)
                # Adjust the rating range for low rated query (from 2.0-3.5 to using median)
                elif is_low_rating_query:
                    median_rating = movies_tfidf_df["bayesian_avg"].median()
                    if bayesian_avg < median_rating:
                        filtered_entity_matches.append(match)

        entity_matches = filtered_entity_matches
        logger.info(
            f"After rating filtering, {len(entity_matches)} entity matches remain"
        )

    # If we have entity matches, boost those recommendations
    final_recommendations = tfidf_recommendations.copy()

    if entity_matches:
        # Create a dictionary of movieId -> entity match score
        entity_match_dict = {
            match["movieId"]: match["match_score"] for match in entity_matches
        }

        # Get highest match score for normalization
        max_match_score = (
            max([match["match_score"] for match in entity_matches])
            if entity_matches
            else 1
        )

        # Update similarity scores for movies that appear in both TF-IDF and entity matches
        for idx, row in tfidf_recommendations.iterrows():
            movie_id = row["movieId"]
            if movie_id in entity_match_dict:
                # Adjust the existing similarity score based on entity match score
                match_score = entity_match_dict[movie_id]
                current_similarity = row["similarity_score"]

                # Normalize match score and apply a stronger, more progressive boost
                normalized_match = match_score / max_match_score
                boost_factor = 0.2 * (
                    normalized_match**0.8
                )  # Progressive boost that favors higher matches

                new_similarity = min(current_similarity + boost_factor, 1.0)
                final_recommendations.at[idx, "similarity_score"] = new_similarity

        # Get IDs from the recommendations we already have
        existing_recommendation_ids = set(final_recommendations["movieId"].values)

        # Find high-scoring entity matches not already included
        # Use a dynamic threshold based on the highest match scores
        threshold_value = max(
            2, max_match_score * 0.25
        )  # At least 2, or 25% of max score
        missing_entity_matches = [
            match
            for match in entity_matches
            if match["movieId"] not in existing_recommendation_ids
            and match["match_score"] >= threshold_value
        ]

        # If we have missing high-scoring matches, add them
        if missing_entity_matches:
            # Take up to 8 top missing matches, more than before
            top_missing = missing_entity_matches[: min(8, len(missing_entity_matches))]

            # Get these movies from the DataFrame
            for match in top_missing:
                movie_data = movies_tfidf_df[
                    movies_tfidf_df["movieId"] == match["movieId"]
                ].iloc[0]

                # Progressive scoring that boosts higher matches more strongly
                normalized_score = match["match_score"] / max_match_score
                base_similarity = 0.5 + (normalized_score * 0.5)  # This scales 0-1

                match_row = pd.DataFrame(
                    {
                        "movieId": [movie_data["movieId"]],
                        "title": [movie_data["title"]],
                        "genres": [movie_data["genres"].split("|")],
                        "bayesian_avg": [movie_data["bayesian_avg"]],
                        "similarity_score": [base_similarity],
                    }
                )
                # Add to recommendations using concat instead of append
                final_recommendations = pd.concat(
                    [final_recommendations, match_row], ignore_index=True
                )

    # Special handling for queries
    if is_low_rating_query:
        # For low rated movies, prioritize by similarity then by lowest rating
        final_recommendations = final_recommendations.sort_values(
            by=["similarity_score", "bayesian_avg"], ascending=[False, True]
        ).head(top_n)
    else:
        # Create a composite ranking that balances similarity, entity matching and quality
        # Normalize ratings to 0-1 range
        min_rating = movies_tfidf_df["bayesian_avg"].min()
        max_rating = movies_tfidf_df["bayesian_avg"].max()
        rating_range = max_rating - min_rating

        final_recommendations["norm_rating"] = (
            final_recommendations["bayesian_avg"] - min_rating
        ) / rating_range

        # Calculate final score with adjusted weights
        final_recommendations["final_score"] = (
            final_recommendations["similarity_score"]
            * 0.8  # Higher weight for similarity
            + final_recommendations["norm_rating"] * 0.2  # Lower weight for rating
        )

        # Sort by the final score
        final_recommendations = final_recommendations.sort_values(
            by="final_score", ascending=False
        ).head(top_n)

        # Remove temporary columns
        final_recommendations = final_recommendations.drop(
            columns=["norm_rating", "final_score"]
        )

    logger.info(f"Returning {len(final_recommendations)} recommendations.")
    return final_recommendations


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

    # Check for rating reference
    rating_threshold = None
    is_low_rating_query = (
        re.search(
            r"\b(?:low|poorly|worst|bad)\s+(?:rated|reviewed|acclaimed)\b",
            prompt.lower(),
        )
        is not None
    )

    if rating_threshold:
        logger.info(f"Detected rating threshold: {rating_threshold}")
    elif is_low_rating_query:
        logger.info("Detected request for low rated movies")

    if not features["processed_text"]:
        logger.warning(
            "No relevant features extracted. Returning top rated movies based on Bayesian Average."
        )
        # Return top N based on precalculated Bayesian average
        if is_low_rating_query:
            # For low rated movies, sort in ascending order
            top_rated = movies_tfidf_df.nsmallest(top_n * 2, "bayesian_avg")[
                ["movieId", "title", "genres", "bayesian_avg"]
            ].copy()
        else:
            top_rated = movies_tfidf_df.nlargest(top_n * 2, "bayesian_avg")[
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
                if is_low_rating_query:
                    more_movies = movies_tfidf_df.nsmallest(top_n * 5, "bayesian_avg")[
                        ["movieId", "title", "genres", "bayesian_avg"]
                    ].copy()
                else:
                    more_movies = movies_tfidf_df.nlargest(top_n * 5, "bayesian_avg")[
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
        top_rated = top_rated.head(top_n)  # Ensure we only return top_n
        return top_rated.assign(
            similarity_score=0.0
        )  # Assign 0 similarity for this case

    # Get improved prompt vector with weighted keywords
    processed_text = features["processed_text"]

    # Add more weight to named entities, especially PERSON, by repeating them
    for entity in features["entities"]:
        # Check if this is a named entity we want to emphasize
        for ent in nlp(entity).ents:
            if ent.label_ in ["PERSON", "WORK_OF_ART", "ORG"]:
                # Add the entity multiple times to increase its weight in TF-IDF
                processed_text = f"{processed_text} {entity} {entity} {entity}"

    # Transform the enhanced prompt
    prompt_vector = tfidf_vectorizer.transform([processed_text])
    similarities = cosine_similarity(prompt_vector, tfidf_feature_matrix).flatten()

    # Get top matches based purely on similarity (get more than needed for filtering)
    filter_multiplier = 2  # Always get more for better candidate pool
    if date_filter_range:
        filter_multiplier *= 3  # Get 3x more results to allow for filtering
    if rating_threshold or is_low_rating_query:
        filter_multiplier *= 2  # Get 2x more results to allow for rating filtering

    # Get more candidates for better diversity
    candidate_indices = np.argsort(similarities)[::-1][: top_n * filter_multiplier]

    # Get the corresponding movie data
    top_recommendations = movies_tfidf_df.iloc[candidate_indices].copy()
    top_recommendations["similarity_score"] = similarities[candidate_indices]

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

    # Apply rating filter if specified
    if rating_threshold:
        # Filter for movies with ratings at or above the threshold
        top_recommendations = top_recommendations[
            top_recommendations["bayesian_avg"] >= rating_threshold
        ]
    elif is_low_rating_query:
        # For low rated movies, prioritize movies with lower ratings
        # Updated threshold: use relative ranking rather than absolute threshold
        median_rating = movies_tfidf_df["bayesian_avg"].median()
        top_recommendations = top_recommendations[
            top_recommendations["bayesian_avg"] < median_rating
        ]
        # Sort by rating ascending (worst first)
        top_recommendations = top_recommendations.sort_values(
            by=["similarity_score", "bayesian_avg"], ascending=[False, True]
        )

    # Create a composite score that balances similarity and rating
    if not is_low_rating_query:
        # Normalize ratings to 0-1 range for better combining with similarity scores
        min_rating = movies_tfidf_df["bayesian_avg"].min()
        max_rating = movies_tfidf_df["bayesian_avg"].max()
        rating_range = max_rating - min_rating

        # Calculate normalized rating
        top_recommendations["norm_rating"] = (
            top_recommendations["bayesian_avg"] - min_rating
        ) / rating_range

        # Calculate composite score with adjusted weights
        top_recommendations["composite_score"] = (
            top_recommendations["similarity_score"]
            * 0.8  # Increased weight for similarity
            + top_recommendations["norm_rating"] * 0.2  # Decreased weight for rating
        )

        # Sort by composite score
        top_recommendations = top_recommendations.sort_values(
            by="composite_score", ascending=False
        )

        # Clean up temporary columns
        top_recommendations = top_recommendations.drop(
            columns=["norm_rating", "composite_score"]
        )

    # Keep only top_n after all filtering
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
