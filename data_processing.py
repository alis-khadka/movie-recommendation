import pandas as pd
import numpy as np
import os
from sentence_transformers import SentenceTransformer
import config


def load_or_compute_llm_data():
    """Loads or computes movie metadata and embeddings for the LLM recommender."""
    if os.path.exists(config.METADATA_PATH) and os.path.exists(config.EMBEDDING_PATH):
        print("Loading precomputed embeddings and metadata...")
        movies_llm = pd.read_parquet(config.METADATA_PATH)
        embeddings = np.load(config.EMBEDDING_PATH)
    else:
        print("Computing embeddings and metadata...")
        movies_llm_raw = pd.read_csv(config.MOVIES_CSV)
        tags_llm_raw = pd.read_csv(config.TAGS_CSV)

        movies_llm_raw["genres"] = (
            movies_llm_raw["genres"].fillna("").apply(lambda g: g.split("|"))
        )

        tags_grouped = (
            tags_llm_raw.groupby("movieId")["tag"]
            .apply(lambda x: list(set(x)))
            .reset_index()
        )
        movies_llm = movies_llm_raw.merge(tags_grouped, on="movieId", how="left")
        movies_llm["tag"] = movies_llm["tag"].apply(
            lambda t: t if isinstance(t, list) else []
        )

        def build_text(row):
            title = row["title"]
            genres = ", ".join(row["genres"])
            tags = ", ".join(row["tag"])
            return f"{title}. Genres: {genres}. Tags: {tags}"

        movies_llm["combined_text"] = movies_llm.apply(build_text, axis=1)

        # Initialize model temporarily just for computation if needed
        embed_model_llm = SentenceTransformer("all-MiniLM-L6-v2")
        movies_llm["embedding"] = movies_llm["combined_text"].apply(
            lambda x: embed_model_llm.encode(x)
        )
        embeddings = np.vstack(movies_llm["embedding"].values)

        movies_llm.drop(columns=["combined_text", "embedding"], inplace=True)
        movies_llm.to_parquet(config.METADATA_PATH, index=False)
        np.save(config.EMBEDDING_PATH, embeddings)
        print("Embeddings and metadata computed and saved.")

    return movies_llm, embeddings


def load_and_preprocess_tfidf_data():
    """Loads and preprocesses data for the TF-IDF/Keyword recommender."""
    print("Loading data for TF-IDF/Keyword recommender...")
    movies_tfidf = pd.read_csv(config.MOVIES_CSV)
    ratings_tfidf = pd.read_csv(config.RATINGS_CSV)

    # Calculate Bayesian Average Rating
    # C = ratings_tfidf.groupby("movieId")["rating"].count().mean() # Original C
    # m = ratings_tfidf["rating"].mean() # Original m
    # Using mean rating of all movies as C (prior mean)
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
    print("Data for TF-IDF/Keyword recommender loaded.")
    return movies_tfidf


# Load data on module import
movies_llm_df, embeddings_matrix = load_or_compute_llm_data()
movies_tfidf_df = load_and_preprocess_tfidf_data()
