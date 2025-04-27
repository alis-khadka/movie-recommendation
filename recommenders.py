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


def parse_date_range(prompt: str):
    """
    Parse date range references like "between 2000-2010", "from 2000 to 2010"
    Returns a tuple of (start_year, end_year) if found, otherwise None.
    """
    # Pattern for "between YYYY-YYYY" or "between YYYY and YYYY"
    between_pattern = r"between\s+(\d{4})[- ](?:and |to )?(\d{4})"
    # Pattern for "from YYYY to YYYY"
    from_to_pattern = r"from\s+(\d{4})\s+to\s+(\d{4})"

    # Check for "between" pattern
    between_match = re.search(between_pattern, prompt.lower())
    if between_match:
        start_year = int(between_match.group(1))
        end_year = int(between_match.group(2))
        return (start_year, end_year)

    # Check for "from-to" pattern
    from_to_match = re.search(from_to_pattern, prompt.lower())
    if from_to_match:
        start_year = int(from_to_match.group(1))
        end_year = int(from_to_match.group(2))
        return (start_year, end_year)

    return None


def parse_multiple_decades(prompt: str):
    """
    Parse multiple decade references like "in 2000s and 2010s"
    Returns a tuple of (min_year, max_year) covering all mentioned decades.
    """
    decades = []

    # Pattern for full decades like "1990s", "2000s" - capture all occurrences
    full_pattern = r"\b(1[0-9]{3}|20[0-9]{2})s\b"
    # Pattern for short decades like "90s", "00s"
    short_pattern = r"\b([1-9]0)s\b"

    # Find all full decade patterns
    full_matches = re.findall(full_pattern, prompt.lower())
    for match in full_matches:
        base_year = int(match)
        decades.append((base_year, base_year + 9))

    # Find all short decade patterns
    short_matches = re.findall(short_pattern, prompt.lower())
    for match in short_matches:
        decade = int(match)
        # Convert "90" to "1990", "00" to "2000", etc.
        if decade < 30:  # Assume 00s, 10s, 20s refer to 2000s, 2010s, 2020s
            base_year = 2000 + decade
        else:  # Assume 30s through 90s refer to 1900s
            base_year = 1900 + decade
        decades.append((base_year, base_year + 9))

    # If multiple decades found, return the min and max years
    if decades:
        min_year = min(decade[0] for decade in decades)
        max_year = max(decade[1] for decade in decades)
        return (min_year, max_year)

    return None


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
                # Only consider reasonable movie years (1900-2025) - Adjust upper bound as needed
                current_year = pd.Timestamp.now().year
                if 1900 <= year <= current_year + 1:
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
        # Check if the genre word itself is present
        if genre.lower() in prompt_words:
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
            # "TITLE", # TITLE is not a standard SpaCy label
        ]:
            entity_dict[ent.text] = ent.label_

    entities = list(entity_dict.keys())
    entity_types = list(entity_dict.values())

    combined_features = list(set(detected_genres + key_phrases + entities))
    processed_text = " ".join(combined_features)

    # Return requested_genres separately to help with genre diversity scoring
    return {
        "genres": detected_genres,
        "keywords": key_phrases,
        "entities": entities,
        "entity_types": entity_types,
        "processed_text": processed_text,
        "requested_genres": detected_genres,  # Store explicitly requested genres
    }


def find_entity_matches(features, top_n=50):
    """Find movies that match extracted entities in titles or tags"""
    entity_matches = []

    # Only process if we have entities
    if not features["entities"]:
        return []

    # Ensure tags column exists and handle potential NaN values
    if "tags" not in movies_tfidf_df.columns:
        logger.warning("Tags column missing, cannot perform entity matching on tags.")
        movies_tfidf_df["tags"] = [
            [] for _ in range(len(movies_tfidf_df))
        ]  # Add empty list if missing
    else:
        # Ensure tags are lists, handle NaN/float types
        movies_tfidf_df["tags"] = movies_tfidf_df["tags"].apply(
            lambda x: (
                x if isinstance(x, list) else ([] if pd.isna(x) else str(x).split("|"))
            )
        )

    # Check entities against movie titles and tags
    # Iterate over a potentially larger set initially if filtering is expected
    initial_check_df = movies_tfidf_df  # Check against all movies initially
    for idx, movie in initial_check_df.iterrows():
        title = movie["title"]
        tags = movie.get("tags", [])  # Use .get for safety
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
    # Return up to top_n matches after sorting all potential matches
    return entity_matches[:top_n]


# --- Recommendation Logic ---


def get_enhanced_recommendations(prompt: str, top_n=10):
    """Generates recommendations with enhanced entity matching"""
    logger.info(f"--- Inside Enhanced Recommender --- V_UPDATED")
    logger.info(f"Received prompt: {prompt}, top_n: {top_n}")
    features = extract_features(prompt)
    logger.info(f"Extracted Features: {features}")

    # Enhanced date range detection - check in priority order
    date_filter_range = None

    # 1. First check for date range format like "between 2000-2010"
    date_range = parse_date_range(prompt)
    if date_range:
        start_year, end_year = date_range
        date_filter_range = (start_year, end_year)
        logger.info(f"Detected date range: between {start_year}-{end_year}")

    # 2. Next check for multiple decades like "2000s and 2010s"
    elif "and" in prompt.lower() and re.search(r"\d+s", prompt.lower()):
        multiple_decades = parse_multiple_decades(prompt)
        if multiple_decades:
            start_year, end_year = multiple_decades
            date_filter_range = (start_year, end_year)
            logger.info(f"Detected multiple decades: {start_year}-{end_year}")

    # 3. Check for single decade reference
    elif not date_filter_range:
        decade_range = parse_decade_reference(prompt)
        if decade_range:
            start_year, end_year = decade_range
            date_filter_range = (start_year, end_year)
            logger.info(
                f"Detected decade reference: {start_year}s ({start_year}-{end_year})"
            )

    # 4. Fall back to specific year extraction from entities
    if not date_filter_range:
        year_range = extract_year_reference(
            features["entities"], features["entity_types"]
        )
        if year_range:
            date_filter_range = year_range
            logger.info(f"Detected specific year reference: {year_range[0]}")

    # Check for rating reference (removed threshold logic, kept low rating flag)
    is_low_rating_query = (
        re.search(
            r"\b(?:low|poorly|worst|bad)\s+(?:rated|reviewed|acclaimed)\b",
            prompt.lower(),
        )
        is not None
    )

    if is_low_rating_query:
        logger.info("Detected request for low rated movies")

    # Get standard TF-IDF recommendations (now includes final_score)
    tfidf_recommendations = get_keyword_recommendations(
        prompt,
        top_n=top_n * 2,  # Get more initially to allow for better merging/sorting
        date_filter_range=date_filter_range,
        is_low_rating_query=is_low_rating_query,
        features=features,
    )

    # Get entity-based matches (check against more movies initially)
    entity_matches = find_entity_matches(features, top_n=100)
    logger.info(f"Found {len(entity_matches)} entity matches initially")

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

    # Filter entity matches by rating if needed (only for low rating query)
    if is_low_rating_query:
        filtered_entity_matches = []
        for match in entity_matches:
            # Look up the movie's rating
            movie_data = movies_tfidf_df[movies_tfidf_df["movieId"] == match["movieId"]]
            if not movie_data.empty:
                bayesian_avg = movie_data["bayesian_avg"].values[0]
                # Define low rating threshold (e.g., below 3.0)
                if bayesian_avg < 3.0:
                    filtered_entity_matches.append(match)
            else:
                logger.warning(
                    f"MovieId {match['movieId']} not found in movies_tfidf_df during rating filter."
                )

        entity_matches = filtered_entity_matches
        logger.info(
            f"After low rating filtering, {len(entity_matches)} entity matches remain"
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
        # Consider matches with score >= 1 (any entity match)
        missing_entity_matches = [
            match
            for match in entity_matches
            if match["movieId"] not in tfidf_recommendation_ids
            # and match["match_score"] >= 3 # Relaxed threshold to include more relevant entity matches
        ]

        if missing_entity_matches:
            # Sort missing matches by score to add the best ones first
            missing_entity_matches.sort(key=lambda x: x["match_score"], reverse=True)

            rows_to_append = []
            added_ids = set(tfidf_recommendations["movieId"].values)
            for match in missing_entity_matches:
                # Skip if needed
                if len(rows_to_append) >= top_n:
                    break
                if match["movieId"] in added_ids:
                    continue

                movie_data_series = movies_tfidf_df[
                    movies_tfidf_df["movieId"] == match["movieId"]
                ]
                if not movie_data_series.empty:
                    movie_data = movie_data_series.iloc[0]

                    # --- Calculate final_score for entity matches ---
                    # Use a base similarity reflecting entity match strength
                    entity_sim_score = 0.5 + min(0.4, match["match_score"] / 10.0)

                    # Normalize rating for this movie
                    bayesian_avg_numeric = pd.to_numeric(
                        movie_data["bayesian_avg"], errors="coerce"
                    )
                    # Use global mean or fallback if needed for normalization consistency
                    # Assuming mean_rating is accessible or recalculate/approximate
                    mean_rating_fallback = 3.0  # Example fallback mean
                    filled_rating = (
                        bayesian_avg_numeric
                        if not pd.isna(bayesian_avg_numeric)
                        else mean_rating_fallback
                    )
                    normalized_rating = (filled_rating / 5.0).clip(
                        0, 1
                    )  # Assuming 5.0 max

                    # Use same weights as in keyword recommender
                    w_sim = 0.7
                    w_rating = 0.3
                    final_score = (entity_sim_score * w_sim) + (
                        normalized_rating * w_rating
                    )
                    # ------------------------------------------------

                    match_row = {
                        "movieId": movie_data["movieId"],
                        "title": movie_data["title"],
                        "genres": (
                            movie_data["genres"].split("|")
                            if isinstance(movie_data["genres"], str)
                            else []
                        ),
                        "bayesian_avg": movie_data["bayesian_avg"],
                        "similarity_score": entity_sim_score,  # Store the entity-based sim score
                        "final_score": final_score,  # Store the combined score
                    }
                    rows_to_append.append(match_row)
                    added_ids.add(match["movieId"])
                else:
                    logger.warning(
                        f"MovieId {match['movieId']} not found when trying to append missing entity match."
                    )

            # Add to recommendations using pd.concat
            if rows_to_append:
                tfidf_recommendations = pd.concat(
                    [tfidf_recommendations, pd.DataFrame(rows_to_append)],
                    ignore_index=True,
                )

            # --- Re-sort combined list by final_score ---
            if is_low_rating_query:
                # Sort by final_score desc, then rating asc
                tfidf_recommendations = tfidf_recommendations.sort_values(
                    by=["final_score", "bayesian_avg"], ascending=[False, True]
                ).head(top_n)
            else:
                # Default sorting by final_score desc
                tfidf_recommendations = tfidf_recommendations.sort_values(
                    by="final_score", ascending=False
                ).head(top_n)
            # --------------------------------------------
        else:  # If no missing entity matches, ensure the original list is sorted correctly
            if is_low_rating_query:
                tfidf_recommendations = tfidf_recommendations.sort_values(
                    by=["final_score", "bayesian_avg"], ascending=[False, True]
                ).head(top_n)
            else:
                tfidf_recommendations = tfidf_recommendations.sort_values(
                    by="final_score", ascending=False
                ).head(top_n)

    # Apply genre diversity enhancement if multiple genres were requested
    if features["requested_genres"] and len(features["requested_genres"]) > 1:
        logger.info(
            f"Applying genre diversity for requested genres: {features['requested_genres']}"
        )
        # Create a more diverse recommendation set
        tfidf_recommendations = enhance_genre_diversity(
            tfidf_recommendations, features["requested_genres"], top_n=top_n
        )

    logger.info(f"Returning {len(tfidf_recommendations)} recommendations.")
    # Return the standard columns, final_score might not be needed in the final API response
    return tfidf_recommendations[
        ["movieId", "title", "genres", "bayesian_avg", "similarity_score"]
    ]


def enhance_genre_diversity(recommendations, requested_genres, top_n=10):
    """
    Enhances genre diversity in the recommendations based on requested genres.
    Ensures that recommendations cover as many of the requested genres as possible.
    """
    if len(requested_genres) <= 1:
        return recommendations.head(top_n)

    # Convert all genre names to lowercase for better matching
    requested_genres_lower = [g.lower() for g in requested_genres]

    # Calculate genre coverage score for each movie
    def calculate_genre_coverage(movie_genres):
        # Convert to lowercase for comparison
        movie_genres_lower = [g.lower() for g in movie_genres]
        # Count how many requested genres appear in this movie
        matched_genres = sum(
            1
            for genre in requested_genres_lower
            if any(genre in mg for mg in movie_genres_lower)
        )
        # Normalize by the total requested genres
        return matched_genres / len(requested_genres_lower)

    # Add genre coverage score
    recommendations["genre_coverage"] = recommendations["genres"].apply(
        calculate_genre_coverage
    )

    # Blend the original score with genre coverage for re-ranking
    # Give more weight to genre coverage for multi-genre queries
    w_original = 0.6
    w_genre_coverage = 0.4

    recommendations["diversity_score"] = (
        recommendations["final_score"] * w_original
        + recommendations["genre_coverage"] * w_genre_coverage
    )

    # Create a diverse set that prioritizes genre coverage
    # First get a base set of recommendations with high genre coverage
    high_coverage = recommendations.sort_values(
        by="genre_coverage", ascending=False
    ).head(min(top_n, len(recommendations)))

    # Then get remaining recommendations based on blended score
    remaining = top_n - len(high_coverage)
    if remaining > 0:
        # Exclude movies already selected
        remaining_recs = recommendations[
            ~recommendations["movieId"].isin(high_coverage["movieId"])
        ]
        # Sort by diversity score
        remaining_recs = remaining_recs.sort_values(
            by="diversity_score", ascending=False
        ).head(remaining)
        # Combine the sets
        diverse_recommendations = pd.concat([high_coverage, remaining_recs])
    else:
        diverse_recommendations = high_coverage

    # Sort final set by diversity score
    return diverse_recommendations.sort_values(
        by="diversity_score", ascending=False
    ).head(top_n)


def get_keyword_recommendations(
    prompt: str,
    top_n=10,
    date_filter_range=None,
    is_low_rating_query=False,
    features=None,
):
    """Generates recommendations based on keyword/TF-IDF matching."""
    logger.info(f"--- Inside Keyword Recommender --- V_UPDATED")
    logger.info(f"Received prompt: {prompt}, top_n: {top_n}")

    # Use pre-extracted features if provided
    if features is None:
        features = extract_features(prompt)
        logger.info(f"Extracted Features (within keyword recommender): {features}")
    else:
        logger.info("Using pre-extracted features.")

    # Date and rating filters are passed in, no need to re-calculate here unless not passed
    if date_filter_range:
        logger.info(f"Applying date filter: {date_filter_range}")
    if is_low_rating_query:
        logger.info("Applying low rating filter.")

    if not features["processed_text"]:
        logger.warning(
            "No relevant features extracted. Returning top rated movies based on Bayesian Average."
        )
        # Base query for top rated movies
        if is_low_rating_query:
            base_query = movies_tfidf_df.sort_values("bayesian_avg", ascending=True)
        else:
            base_query = movies_tfidf_df.sort_values("bayesian_avg", ascending=False)

        # Apply date filter if specified
        if date_filter_range:
            start_year, end_year = date_filter_range
            # Extract year from title (assuming format includes year in parentheses)
            # Handle potential errors during extraction/conversion
            base_query["year"] = base_query["title"].str.extract(
                r"\((\d{4})\)", expand=False
            )
            base_query["year"] = pd.to_numeric(base_query["year"], errors="coerce")
            base_query = base_query.dropna(
                subset=["year"]
            )  # Drop rows where year couldn't be extracted
            base_query["year"] = base_query["year"].astype(int)

            base_query = base_query[
                (base_query["year"] >= start_year) & (base_query["year"] <= end_year)
            ]

        top_rated = base_query.head(top_n)[
            ["movieId", "title", "genres", "bayesian_avg"]
        ].copy()

        top_rated["genres"] = top_rated["genres"].apply(
            lambda x: x.split("|") if isinstance(x, str) else []
        )
        return top_rated.assign(
            similarity_score=0.0, final_score=0.0
        )  # Assign 0 similarity for this case

    prompt_vector = tfidf_vectorizer.transform([features["processed_text"]])
    similarities = cosine_similarity(prompt_vector, tfidf_feature_matrix).flatten()

    # Create a DataFrame with similarities and movie data
    similarity_df = pd.DataFrame(
        {
            "movieId": movies_tfidf_df["movieId"],
            "title": movies_tfidf_df["title"],
            "genres": movies_tfidf_df["genres"],
            "bayesian_avg": movies_tfidf_df["bayesian_avg"],
            "similarity_score": similarities,
        }
    )

    # --- Add Rating Normalization and Final Score Calculation ---
    w_sim = 0.7
    w_rating = 0.3
    max_rating = 5.0  # Assuming 0-5 scale

    # Convert bayesian_avg to numeric, handle errors, fill NaNs
    similarity_df["bayesian_avg_numeric"] = pd.to_numeric(
        similarity_df["bayesian_avg"], errors="coerce"
    )
    mean_rating = similarity_df["bayesian_avg_numeric"].mean()
    similarity_df["bayesian_avg_numeric"] = similarity_df[
        "bayesian_avg_numeric"
    ].fillna(mean_rating if not pd.isna(mean_rating) else max_rating / 2)

    # Normalize rating
    similarity_df["normalized_rating"] = (
        similarity_df["bayesian_avg_numeric"] / max_rating
    ).clip(0, 1)

    # Calculate final score
    similarity_df["final_score"] = (similarity_df["similarity_score"] * w_sim) + (
        similarity_df["normalized_rating"] * w_rating
    )
    # ----------------------------------------------------------

    # Calculate genre diversity score if multiple genres were requested
    if "requested_genres" in features and len(features["requested_genres"]) > 1:
        # Convert genres strings to lists first
        similarity_df["genres_list"] = similarity_df["genres"].apply(
            lambda x: x.split("|") if isinstance(x, str) else []
        )

        # Calculate genre match score - how many of the requested genres are in the movie
        requested_genres_lower = [g.lower() for g in features["requested_genres"]]

        def calc_genre_match(genres_list):
            genres_lower = [g.lower() for g in genres_list]
            matched = sum(
                1 for rg in requested_genres_lower if any(rg in g for g in genres_lower)
            )
            # Normalize by the total requested genres
            return matched / len(requested_genres_lower)

        similarity_df["genre_match_score"] = similarity_df["genres_list"].apply(
            calc_genre_match
        )

        # Incorporate genre match into final score
        similarity_df["final_score"] = (similarity_df["final_score"] * 0.7) + (
            similarity_df["genre_match_score"] * 0.3
        )

        # Clean up temporary column
        similarity_df = similarity_df.drop(columns=["genres_list"])

    # Apply date filter if specified
    if date_filter_range:
        start_year, end_year = date_filter_range
        # Extract year from title
        similarity_df["year"] = similarity_df["title"].str.extract(
            r"\((\d{4})\)", expand=False
        )
        similarity_df["year"] = pd.to_numeric(similarity_df["year"], errors="coerce")
        similarity_df = similarity_df.dropna(subset=["year"])
        similarity_df["year"] = similarity_df["year"].astype(int)

        similarity_df = similarity_df[
            (similarity_df["year"] >= start_year) & (similarity_df["year"] <= end_year)
        ]
        if "year" in similarity_df.columns:
            similarity_df = similarity_df.drop(columns=["year"], errors="ignore")

    # Apply rating filter if specified (only for low rating query)
    if is_low_rating_query:
        # Filter for movies with low ratings
        similarity_df = similarity_df[similarity_df["bayesian_avg_numeric"] < 1.5]

    # Sort based on query type using the new final_score
    if is_low_rating_query:
        # Sort by final_score descending (still want relevant low-rated), then rating ascending (worst first)
        top_recommendations = similarity_df.sort_values(
            by=["final_score", "bayesian_avg_numeric"], ascending=[False, True]
        )
    else:
        # Default: Sort by final_score descending
        top_recommendations = similarity_df.sort_values(
            by="final_score", ascending=False
        )

    # Keep only top_n after all filtering and sorting
    top_recommendations = top_recommendations.head(top_n)

    # Convert genres string to list
    top_recommendations["genres"] = top_recommendations["genres"].apply(
        lambda x: x.split("|") if isinstance(x, str) else []
    )

    logger.info(
        f"Returning {len(top_recommendations)} recommendations based on combined score."
    )

    # Return relevant columns, including final_score if needed, or just the original required ones
    # Let's keep the output schema consistent for now
    return top_recommendations[
        [
            "movieId",
            "title",
            "genres",
            "bayesian_avg",
            "similarity_score",
            "final_score",
        ]
    ]
