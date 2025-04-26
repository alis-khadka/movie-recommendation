import pandas as pd
import numpy as np
import os

# from sentence_transformers import SentenceTransformer # No longer needed here
import config
import ast
import logging
import re

logger = logging.getLogger(__name__)

# Cache for the embeddings matrix - initialize as None for lazy loading
_embeddings_matrix_cache = None


def safe_literal_eval(val):
    """Safely evaluate string representations of lists/arrays, or convert numpy arrays."""
    if isinstance(val, list):
        return val  # Already a list

    if isinstance(val, np.ndarray):
        # If pandas somehow loads it as a proper numpy array object
        return val.tolist()

    if isinstance(val, str):
        cleaned_val = val.strip()
        # Attempt 1: Handle numpy array string representation "['a' 'b']"
        # Use regex to find single quotes separated by space(s) and replace with comma-space
        if cleaned_val.startswith("[") and cleaned_val.endswith("]"):
            # More robust regex: find a closing quote, optional spaces, an opening quote
            # Replace with closing quote, comma, space, opening quote
            corrected_string = re.sub(r"(?<=\')\s+(?=\')", "', '", cleaned_val)
            try:
                # Now try literal_eval on the corrected string
                evaluated = ast.literal_eval(corrected_string)
                return evaluated if isinstance(evaluated, list) else []
            except (ValueError, SyntaxError, TypeError) as e:
                logger.warning(
                    f"Could not parse numpy-like string after correction: {corrected_string} (Original: {val}). Error: {e}"
                )
                # Fall through if correction or parsing fails

        # Attempt 2: Try standard literal_eval for regular list strings "['a', 'b']"
        try:
            evaluated = ast.literal_eval(val)
            return evaluated if isinstance(evaluated, list) else []
        except (ValueError, SyntaxError, TypeError):
            if val == "(no genres listed)":  # Handle specific known non-list strings
                return []
            # Don't log a warning here if the first attempt failed, avoid double logging
            if not (cleaned_val.startswith("[") and cleaned_val.endswith("]")):
                logger.warning(
                    f"Could not parse string to list using literal_eval: {val}"
                )
            return []  # Cannot parse string

    if pd.isna(val):
        return []

    logger.warning(
        f"Unexpected type encountered in safe_literal_eval: {type(val)}, value: {val}"
    )
    return []  # Default to empty list


def load_llm_data_from_parquet():
    """Loads movie metadata and embeddings strictly from precomputed files."""
    if not os.path.exists(config.METADATA_PATH) or not os.path.exists(
        config.EMBEDDING_PATH
    ):
        logger.error(
            f"Required files not found: {config.METADATA_PATH} or {config.EMBEDDING_PATH}"
        )
        raise FileNotFoundError(
            f"Could not find precomputed metadata or embeddings. Please ensure '{config.METADATA_PATH}' and '{config.EMBEDDING_PATH}' exist."
        )

    logger.info(
        f"Loading precomputed data from {config.METADATA_PATH} and {config.EMBEDDING_PATH}..."
    )
    movies_llm = pd.read_parquet(config.METADATA_PATH)
    print("--- Data types directly from Parquet --- ")  # Add print statement
    print(movies_llm.dtypes)  # Add print statement
    print("----------------------------------------")  # Add print statement

    # --- Explicitly convert columns after loading ---
    if not movies_llm.empty:
        logger.info(
            "Applying type conversion to 'genres' and 'tag' columns after loading parquet."
        )
        movies_llm["genres"] = movies_llm["genres"].apply(safe_literal_eval)

        # Check if 'tag' column exists
        has_tag_column = "tag" in movies_llm.columns

        # If tag column exists, check if it has non-empty values
        has_tags = False
        if has_tag_column:
            # Convert first to be safe
            movies_llm["tag"] = movies_llm["tag"].apply(safe_literal_eval)
            # Check if any movie has tags
            has_tags = any(len(tags) > 0 for tags in movies_llm["tag"])
            logger.info(f"Parquet file has 'tag' column with data: {has_tags}")

            if not has_tags:
                logger.warning(
                    "All tags in parquet file are empty. Will reload from CSV."
                )

        # Always load tags from CSV if column doesn't exist or all tags are empty
        if not has_tag_column or not has_tags:
            logger.info("Loading tags from CSV...")
            # Load tags from CSV and merge them
            movies_llm = add_tags_from_csv(movies_llm)

        # Log type after conversion for verification
        if not movies_llm.empty:
            logger.info(
                f"Type of 'genres' column after conversion: {movies_llm['genres'].dtype}, Sample: {movies_llm['genres'].iloc[0] if len(movies_llm) > 0 else 'N/A'}"
            )
            logger.info(
                f"Type of 'tag' column after conversion: {movies_llm['tag'].dtype}, Sample: {movies_llm['tag'].iloc[0] if len(movies_llm) > 0 else 'N/A'}"
            )
    else:
        logger.warning(f"Loaded DataFrame from {config.METADATA_PATH} is empty.")

    logger.info("Precomputed LLM data loaded successfully.")
    return movies_llm


def add_tags_from_csv(movies_df):
    """Add tags from the original CSV file to the movies dataframe."""
    logger.info(f"Loading tags from CSV file: {config.TAGS_CSV}")

    try:
        # Load tags from CSV
        tags_df = pd.read_csv(config.TAGS_CSV)

        # Log some information about the loaded tags
        logger.info(f"Loaded {len(tags_df)} tag entries from CSV")
        logger.info(f"Tags for {tags_df['movieId'].nunique()} unique movies")

        # Log sample tags for debugging
        sample_tags = tags_df.head(10)
        logger.info(f"Sample tags data:\n{sample_tags}")

        # Group tags by movieId
        grouped_tags = tags_df.groupby("movieId")["tag"].apply(list).reset_index()
        logger.info(f"After grouping, we have tags for {len(grouped_tags)} movies")

        # Sample the grouped tags for debugging
        if not grouped_tags.empty:
            sample_grouped = grouped_tags.head(3)
            for _, row in sample_grouped.iterrows():
                logger.info(f"Movie ID {row['movieId']} has tags: {row['tag']}")

        # Create a dictionary for quick lookup (movieId -> tag list)
        movie_tags_dict = dict(zip(grouped_tags["movieId"], grouped_tags["tag"]))

        # See if the movieIds in our dataframe match the ones in the tags
        common_ids = set(movies_df["movieId"]).intersection(
            set(grouped_tags["movieId"])
        )
        logger.info(f"Movies with matching IDs in both dataframes: {len(common_ids)}")

        # Add tags to movies dataframe
        movies_df["tag"] = movies_df["movieId"].apply(
            lambda x: movie_tags_dict.get(x, [])
        )

        # Log results
        with_tags = sum(len(tags) > 0 for tags in movies_df["tag"])
        logger.info(
            f"Added tags to {with_tags} movies out of {len(movies_df)} total movies"
        )

        # Log a few examples of movies with tags
        movies_with_tags = movies_df[movies_df["tag"].apply(lambda x: len(x) > 0)].head(
            5
        )
        for _, row in movies_with_tags.iterrows():
            logger.info(
                f"Movie '{row['title']}' (ID: {row['movieId']}) has tags: {row['tag']}"
            )

        return movies_df

    except Exception as e:
        logger.error(f"Error loading tags from CSV: {str(e)}")
        logger.exception("Stack trace:")
        # If there's an error, add an empty list for each movie
        movies_df["tag"] = [[] for _ in range(len(movies_df))]
        return movies_df


def get_embeddings_matrix():
    """Lazy-loads the embeddings matrix only when needed."""
    global _embeddings_matrix_cache

    if _embeddings_matrix_cache is None:
        logger.info(f"Loading embeddings matrix from {config.EMBEDDING_PATH}...")
        _embeddings_matrix_cache = np.load(config.EMBEDDING_PATH)
        logger.info(f"Embeddings matrix loaded: shape {_embeddings_matrix_cache.shape}")

    return _embeddings_matrix_cache


def load_and_preprocess_tfidf_data():
    """Loads and preprocesses data for the TF-IDF/Keyword recommender."""
    print("Loading data for TF-IDF/Keyword recommender...")
    movies_tfidf = pd.read_csv(config.MOVIES_CSV)
    ratings_tfidf = pd.read_csv(config.RATINGS_CSV)

    # Calculate Bayesian Average Rating
    C = ratings_tfidf["rating"].mean()
    # Using mean number of ratings per movie as m (prior count strength)
    m = ratings_tfidf.groupby("movieId")["rating"].count().mean()

    bayesian_stats = (
        ratings_tfidf.groupby("movieId")
        .agg(num_ratings=("rating", "count"), sum_ratings=("rating", "sum"))
        .reset_index()
    )
    # Original formula: (C * m + bayesian_stats['sum_ratings']) / (C + bayesian_stats['num_ratings'])
    # Applying with C=mean_rating, m=mean_count
    bayesian_stats["bayesian_avg"] = (m * C + bayesian_stats["sum_ratings"]) / (
        m + bayesian_stats["num_ratings"]
    )

    movies_tfidf = movies_tfidf.merge(
        bayesian_stats[["movieId", "bayesian_avg"]], on="movieId", how="left"
    )
    # Fill missing Bayesian averages with the global mean rating C
    movies_tfidf["bayesian_avg"] = movies_tfidf["bayesian_avg"].fillna(C)

    movies_tfidf["processed_genres"] = (
        movies_tfidf["genres"].str.replace("|", " ", regex=False).fillna("")
    )

    # Also load and add tags to the TF-IDF version for consistency
    try:
        tags_df = pd.read_csv(config.TAGS_CSV)
        # Group tags by movieId
        grouped_tags = tags_df.groupby("movieId")["tag"].apply(list).reset_index()
        # Create a dictionary for quick lookup (movieId -> tag list)
        movie_tags_dict = dict(zip(grouped_tags["movieId"], grouped_tags["tag"]))
        # Add tags to movies dataframe
        movies_tfidf["tags"] = movies_tfidf["movieId"].apply(
            lambda x: movie_tags_dict.get(x, [])
        )
    except Exception as e:
        print(f"Warning: Could not load tags for TF-IDF recommender: {e}")
        movies_tfidf["tags"] = [[] for _ in range(len(movies_tfidf))]

    print("Data for TF-IDF/Keyword recommender loaded.")
    return movies_tfidf


# Load data on module import
# Use the new function dedicated to loading from parquet
movies_llm_df = load_llm_data_from_parquet()
# Keep loading TF-IDF data as before, unless specified otherwise
movies_tfidf_df = load_and_preprocess_tfidf_data()

# Instead of using property, we'll set this to None
# This value will be replaced by the function call when needed
embeddings_matrix = None
