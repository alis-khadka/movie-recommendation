import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import traceback
from fastapi import HTTPException
import logging
import time
from google.api_core import exceptions as api_exceptions
from google.genai import types

from schemas import LLMQuery, RecommendationQuery
from data_processing import movies_llm_df, get_embeddings_matrix
from models import gemini_client, embed_model
from recommenders import get_keyword_recommendations, get_enhanced_recommendations
import pandas as pd  # Add pandas import
import os  # Add os import

logger = logging.getLogger(__name__)

# --- Add logging right after import ---
logger.info(f"DataFrame movies_llm_df loaded. Info:")
logger.info(movies_llm_df.info())
logger.info(f"First 5 rows of genres:\n{movies_llm_df['genres'].head()}")
logger.info(f"First 5 rows of tags:\n{movies_llm_df['tag'].head()}")
logger.info(f"Data type of 'genres' column: {movies_llm_df['genres'].dtype}")
logger.info(f"Data type of 'tag' column: {movies_llm_df['tag'].dtype}")
# --- End of added logging ---


def generate_llm_recommendations(query: LLMQuery):
    """Handles the logic for generating LLM-based recommendations."""
    try:
        logger.info(f"--- Inside LLM Controller Logic ---")
        logger.info(
            # Use query.top_n for logging
            f"Processing input: {query.user_input}, top_n: {query.top_n}"
        )

        # Get embeddings matrix only when needed
        embeddings_matrix = get_embeddings_matrix()

        query_vec = embed_model.encode(query.user_input)
        similarities = cosine_similarity([query_vec], embeddings_matrix)[0]
        # Use query.top_n to fetch candidates
        num_candidates = min(query.top_n, len(movies_llm_df))
        top_indices = np.argsort(similarities)[::-1][:num_candidates]
        top_movies = movies_llm_df.iloc[top_indices]
        logger.info(f"Found {len(top_movies)} candidates based on similarity.")

        # Correctly unpack the tuple from iterrows() when using enumerate
        movie_list = "\n".join(
            f"{i+1}. {row_data['title']} (Genres: {', '.join(row_data['genres'] if isinstance(row_data['genres'], list) else [])}; Tags: {', '.join(row_data['tag'] if isinstance(row_data['tag'], list) else [])})"  # i is the 0-based counter from enumerate
            # index is the original DataFrame index from iterrows
            # row_data is the Pandas Series for the row
            for i, (index, row_data) in enumerate(top_movies.iterrows())
        )

        prompt = f"""
User input: {query.user_input}

Please recommend the {query.top_n} most relevant movies from along with their genres and tags only included in the movielens dataset (small). The response should be in the following format and don't include any extra information:

Movie Title
Genres
Tags

Examples:

Bag Man, The (2014)
Crime|Drama|Thriller
mystery

Fracture (2007)
Crime|Drama|Mystery|Thriller
courtroom drama|twist ending

"""
        logger.info("Sending prompt to Gemini...")

        # Add retry logic for API calls
        max_retries = 3
        retry_delay = 1  # seconds

        for attempt in range(max_retries):
            try:
                response = gemini_client.models.generate_content(
                    model="models/gemini-2.0-flash",
                    contents=prompt,
                    config={
                        "temperature": 0,
                        "system_instruction": "You are a helpful movie assistant. Your response should not include any extra information, explanation or pretext.",
                    },
                )
                break  # If successful, break out of retry loop
            except (
                api_exceptions.ResourceExhausted,
                api_exceptions.ServiceUnavailable,
            ) as e:
                if attempt < max_retries - 1:
                    logger.warning(
                        f"Gemini API error: {e}. Retrying in {retry_delay}s (attempt {attempt+1}/{max_retries})"
                    )
                    time.sleep(retry_delay * (2**attempt))  # Exponential backoff
                else:
                    logger.error(f"Gemini API failed after {max_retries} attempts: {e}")
                    raise HTTPException(
                        status_code=503,
                        detail=f"External AI service temporarily unavailable. Please try again later.",
                    )
            except Exception as e:
                logger.error(f"Unexpected error calling Gemini API: {e}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Error communicating with recommendation service: {str(e)}",
                )

        recommendation_text = ""
        try:
            # Simplified response parsing to match original snippet
            if hasattr(response, "text"):
                recommendation_text = response.text.strip()
            else:
                logger.warning(
                    "Warning: Could not extract text from Gemini response using .text."
                )
                recommendation_text = (
                    "Could not parse recommendation from LLM response."
                )
            logger.info(f"Gemini Response: {recommendation_text}")
        except Exception as e:
            logger.error(f"Error accessing Gemini response text: {e}")
            logger.error(f"Full Gemini Response object: {response}")
            recommendation_text = "Could not generate recommendations from the LLM."

        # Parse the recommendation text
        parsed_recommendations = []
        if recommendation_text and "Could not" not in recommendation_text:
            movie_blocks = recommendation_text.strip().split("\n\n")
            for block in movie_blocks:
                lines = block.strip().split("\n")
                if len(lines) >= 3:
                    title = lines[0].strip()
                    genres = (
                        [g.strip() for g in lines[1].split("|")]
                        if "|" in lines[1]
                        else [lines[1].strip()]
                    )
                    tags = (
                        [t.strip() for t in lines[2].split("|")]
                        if "|" in lines[2]
                        else [lines[2].strip()]
                    )
                    parsed_recommendations.append(
                        {"title": title, "genres": genres, "tags": tags}
                    )
                elif len(lines) == 2:  # Handle cases with missing tags maybe?
                    title = lines[0].strip()
                    genres = (
                        [g.strip() for g in lines[1].split("|")]
                        if "|" in lines[1]
                        else [lines[1].strip()]
                    )
                    parsed_recommendations.append(
                        {"title": title, "genres": genres, "tags": []}
                    )

        candidates_output = []
        for _, row in top_movies.iterrows():
            title = row.get("title", "Unknown Title")  # Safer title access

            # Process Genres
            raw_genres = row.get("genres")  # Get raw value first
            if isinstance(raw_genres, list):
                # Handle the specific [''] case which resulted from split('|') on an empty string during data processing
                if raw_genres == [""]:
                    genres = []
                else:
                    genres = (
                        raw_genres  # It's a valid list (or potentially empty list [])
                    )
            else:
                # Handle NaN, None, or other non-list types
                genres = []

            # Process Tags (using the 'tag' column from the DataFrame)
            raw_tags = row.get("tag")  # Get raw value from 'tag' column
            if isinstance(raw_tags, list):
                # Assuming tags don't have the [''] issue like genres might
                tags = raw_tags  # It's a valid list (or potentially empty list [])
            else:
                # Handle NaN, None, or other non-list types
                tags = []

            # Add debug logging to inspect values
            logger.debug(
                f"Processing candidate: {title}, Raw Genres: {raw_genres}, Processed Genres: {genres}, Raw Tags: {raw_tags}, Processed Tags: {tags}"
            )

            candidates_output.append(
                {
                    "title": title,
                    "genres": genres,  # Use the processed genres list
                    "tags": tags,  # Use the processed tags list (output key is 'tags')
                }
            )

        return {
            "user_input": query.user_input,
            "raw_recommendations": recommendation_text,  # Renamed for clarity
            "parsed_recommendations": parsed_recommendations,
            "candidates": candidates_output,
        }

    except HTTPException as http_exc:
        # Re-raise HTTP exceptions without modification
        raise http_exc
    except Exception as e:
        logger.exception(f"Error in LLM controller: {e}")
        traceback.print_exc()
        # Re-raise as HTTPException for the router to catch
        raise HTTPException(
            status_code=500, detail=f"An internal error occurred in LLM logic: {str(e)}"
        )


def handle_recommendation_request(query: RecommendationQuery):
    """Handles the logic for generating keyword-based recommendations."""
    try:
        recommendations = get_enhanced_recommendations(query.prompt, top_n=query.top_n)

        # Convert recommendations to a list of dictionaries for response
        formatted_recommendations = []
        for _, row in recommendations.iterrows():
            formatted_recommendations.append(
                {
                    "movieId": int(row["movieId"]),
                    "title": row["title"],
                    "genres": row["genres"],
                    "similarity_score": float(row["similarity_score"]),
                }
            )

        return {"query": query.prompt, "recommendations": formatted_recommendations}
    except Exception as e:
        print(traceback.format_exc())
        raise HTTPException(
            status_code=500, detail=f"Error generating recommendations: {str(e)}"
        )


def create_movie_tuning_dataset():
    """Creates a tuning dataset from the MovieLens small dataset.

    Fine-tuning dataset limitations for Gemini 1.5 Flash:
    - Maximum input size per example: 40,000 characters
    - Maximum output size per example: 5,000 characters
    - Only input-output pair examples are supported (chat-style conversations not supported)
    """
    logger.info("Creating tuning dataset from MovieLens dataset")

    # Load the movies.csv file
    movies_path = os.path.join(os.getcwd(), "ml-latest-small", "movies.csv")
    movies_df = pd.read_csv(movies_path)

    # Load the ratings.csv file
    ratings_path = os.path.join(os.getcwd(), "ml-latest-small", "ratings.csv")
    ratings_df = pd.read_csv(ratings_path)

    # Load the tags.csv file
    tags_path = os.path.join(os.getcwd(), "ml-latest-small", "tags.csv")
    tags_df = pd.read_csv(tags_path)

    # Aggregate ratings by movieId to get average rating
    movie_ratings = (
        ratings_df.groupby("movieId")["rating"].agg(["mean", "count"]).reset_index()
    )
    movie_ratings = movie_ratings[
        movie_ratings["count"] > 10
    ]  # Only use movies with more than 10 ratings

    # Aggregate tags by movieId
    tags_agg = (
        tags_df.groupby("movieId")["tag"]
        .apply(lambda x: "|".join(x.unique()))
        .reset_index()
    )

    # Merge with movies data
    tuning_data = pd.merge(movies_df, movie_ratings, on="movieId")
    tuning_data = pd.merge(tuning_data, tags_agg, on="movieId", how="left")
    tuning_data["tag"].fillna(
        "popular", inplace=True
    )  # Default tag for movies without tags

    # Order by rating (highest first)
    tuning_data = tuning_data.sort_values(by="mean", ascending=False)

    # Extract year from title and create decade column
    tuning_data["year"] = tuning_data["title"].str.extract(r"\((\d{4})\)").astype(float)
    tuning_data["decade"] = (tuning_data["year"] // 10 * 10).astype(int)

    # Create input-output pairs for tuning
    tuning_examples = []

    # Create examples for genre-based recommendations
    genres = tuning_data["genres"].str.split("|").explode().unique()
    for genre in genres:
        # Find top rated movies for each genre
        genre_movies = tuning_data[tuning_data["genres"].str.contains(genre)]
        if len(genre_movies) >= 5:
            top_movies = genre_movies.head(5)

            input_text = f"Recommend me {genre} movies"

            # Structure output to match our expected format in recommendations
            output_lines = []
            for _, movie in top_movies.iterrows():
                output_lines.append(movie["title"])
                output_lines.append(movie["genres"])
                output_lines.append(movie["tag"])
                output_lines.append("")  # Empty line between entries

            output_text = "\n".join(output_lines).strip()

            # Ensure we're within the character limits for fine-tuning
            if len(input_text) <= 40000 and len(output_text) <= 5000:
                tuning_examples.append((input_text, output_text))

    # Create examples for decade-based recommendations
    for decade in sorted(tuning_data["decade"].unique()):
        if not np.isnan(decade):
            decade_movies = tuning_data[tuning_data["decade"] == decade]
            if len(decade_movies) >= 5:
                top_movies = decade_movies.head(5)

                input_text = f"Recommend movies from the {int(decade)}s"

                # Structure output
                output_lines = []
                for _, movie in top_movies.iterrows():
                    output_lines.append(movie["title"])
                    output_lines.append(movie["genres"])
                    output_lines.append(movie["tag"])
                    output_lines.append("")  # Empty line between entries

                output_text = "\n".join(output_lines).strip()

                if len(input_text) <= 40000 and len(output_text) <= 5000:
                    tuning_examples.append((input_text, output_text))

    # Create examples for combined genre and decade queries
    for genre in genres[
        :10
    ]:  # Limit to first 10 genres to keep dataset size reasonable
        for decade in sorted(tuning_data["decade"].unique())[-4:]:  # Last 4 decades
            if not np.isnan(decade):
                genre_decade_movies = tuning_data[
                    (tuning_data["genres"].str.contains(genre))
                    & (tuning_data["decade"] == decade)
                ]
                if len(genre_decade_movies) >= 3:
                    top_movies = genre_decade_movies.head(3)

                    input_text = f"Recommend {genre} movies from the {int(decade)}s"

                    # Structure output
                    output_lines = []
                    for _, movie in top_movies.iterrows():
                        output_lines.append(movie["title"])
                        output_lines.append(movie["genres"])
                        output_lines.append(movie["tag"])
                        output_lines.append("")  # Empty line between entries

                    output_text = "\n".join(output_lines).strip()

                    if len(input_text) <= 40000 and len(output_text) <= 5000:
                        tuning_examples.append((input_text, output_text))

    # Create examples for rating-based recommendations
    rating_thresholds = [4.5, 4.0, 3.5]
    for threshold in rating_thresholds:
        high_rated = tuning_data[tuning_data["mean"] >= threshold].head(5)
        if len(high_rated) >= 5:
            input_text = f"Recommend highly rated movies"

            output_lines = []
            for _, movie in high_rated.iterrows():
                output_lines.append(movie["title"])
                output_lines.append(movie["genres"])
                output_lines.append(movie["tag"])
                output_lines.append("")  # Empty line between entries

            output_text = "\n".join(output_lines).strip()

            if len(input_text) <= 40000 and len(output_text) <= 5000:
                tuning_examples.append((input_text, output_text))

    # Create examples for tag-based recommendations
    # Get the most common tags
    common_tags = tags_df["tag"].value_counts().head(15).index.tolist()

    for tag in common_tags:
        # Find movies with this tag
        tag_movie_ids = tags_df[tags_df["tag"] == tag]["movieId"].unique()
        tag_movies = tuning_data[tuning_data["movieId"].isin(tag_movie_ids)]

        if len(tag_movies) >= 3:
            top_tag_movies = tag_movies.head(3)

            input_text = f"Recommend movies with {tag}"

            output_lines = []
            for _, movie in top_tag_movies.iterrows():
                output_lines.append(movie["title"])
                output_lines.append(movie["genres"])
                output_lines.append(movie["tag"])
                output_lines.append("")  # Empty line between entries

            output_text = "\n".join(output_lines).strip()

            if len(input_text) <= 40000 and len(output_text) <= 5000:
                tuning_examples.append((input_text, output_text))

    # Check and log statistics about the tuning dataset
    input_lengths = [len(i) for i, _ in tuning_examples]
    output_lengths = [len(o) for _, o in tuning_examples]

    logger.info(f"Created {len(tuning_examples)} tuning examples")
    logger.info(
        f"Input length stats - Min: {min(input_lengths)}, Max: {max(input_lengths)}, Avg: {sum(input_lengths)/len(input_lengths)}"
    )
    logger.info(
        f"Output length stats - Min: {min(output_lengths)}, Max: {max(output_lengths)}, Avg: {sum(output_lengths)/len(output_lengths)}"
    )

    # Verify all examples are within limits
    valid_examples = [
        (i, o) for i, o in tuning_examples if len(i) <= 40000 and len(o) <= 5000
    ]
    if len(valid_examples) < len(tuning_examples):
        logger.warning(
            f"Removed {len(tuning_examples) - len(valid_examples)} examples exceeding character limits"
        )
        tuning_examples = valid_examples

    # Convert to GenerativeAI TuningDataset
    tuning_dataset = types.TuningDataset(
        examples=[
            types.TuningExample(
                text_input=i,
                output=o,
            )
            for i, o in tuning_examples
        ],
    )

    return tuning_dataset


def tune_movie_recommendation_model():
    """Creates and tunes a movie recommendation model using the MovieLens dataset.

    Tuned model limitations for Gemini 1.5 Flash:
    - Input limit of a tuned model is 40,000 characters
    - JSON mode is not supported with tuned models
    - Only text input is supported
    """
    logger.info("Starting model tuning process")

    try:
        # Create the tuning dataset
        tuning_dataset = create_movie_tuning_dataset()

        # Start the tuning job
        tuning_job = gemini_client.tunings.tune(
            base_model="models/gemini-1.5-flash-001-tuning",
            training_dataset=tuning_dataset,
            config=types.CreateTuningJobConfig(
                epoch_count=5,
                batch_size=4,
                learning_rate=0.001,
                tuned_model_display_name="movie_recommendation_model",
            ),
        )

        logger.info(f"Tuning job created with ID: {tuning_job.name}")
        return tuning_job

    except Exception as e:
        logger.exception(f"Error during model tuning: {e}")
        raise


def generate_tuned_recommendations(query: RecommendationQuery, model_name):
    """Generates movie recommendations using a tuned model."""
    logger.info(f"Generating recommendations with tuned model: {model_name}")

    try:
        # For tuned models, we need to incorporate the system instruction into the prompt
        # since system_instruction parameter isn't supported
        enhanced_prompt = f"""You are a helpful movie assistant. Your response should not include any extra information, explanation or pretext.

User input: {query.prompt}

Please recommend the {query.top_n} most relevant movies from along with their genres and tags only. The response should be in the following format and don't include any extra information:

Movie Title
Genres
Tags

Examples:

Bag Man, The (2014)
Crime|Drama|Thriller
mystery

Fracture (2007)
Crime|Drama|Mystery|Thriller
courtroom drama|twist ending"""

        # Generate content with the tuned model
        response = gemini_client.models.generate_content(
            model=model_name,
            contents=enhanced_prompt,
            config={
                "temperature": 0.1,
                "max_output_tokens": 1024,
            },
        )

        # Parse the response
        parsed_recommendations = []
        if response.text and "Could not" not in response.text:
            movie_blocks = response.text.strip().split("\n\n")
            for block in movie_blocks:
                lines = block.strip().split("\n")
                if len(lines) >= 3:
                    title = lines[0].strip()
                    genres = (
                        [g.strip() for g in lines[1].split("|")]
                        if "|" in lines[1]
                        else [lines[1].strip()]
                    )
                    tags = (
                        [t.strip() for t in lines[2].split("|")]
                        if "|" in lines[2]
                        else [lines[2].strip()]
                    )
                    parsed_recommendations.append(
                        {
                            "title": title,
                            "genres": genres,
                            "tags": tags,
                            "similarity_score": 1.0,
                        }
                    )
                elif len(lines) == 2:  # Handle cases with missing tags
                    title = lines[0].strip()
                    genres = (
                        [g.strip() for g in lines[1].split("|")]
                        if "|" in lines[1]
                        else [lines[1].strip()]
                    )
                    parsed_recommendations.append(
                        {
                            "title": title,
                            "genres": genres,
                            "tags": [],
                            "similarity_score": 1.0,
                        }
                    )

        return {
            "query": query.prompt,
            "raw_recommendations": response.text.strip() if response.text else "",
            "recommendations": parsed_recommendations,
        }

    except Exception as e:
        logger.exception(f"Error generating tuned recommendations: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error generating recommendations with tuned model: {str(e)}",
        )


def list_available_models():
    """Lists all available models including tuned models."""
    logger.info("Listing available models")

    try:
        models = []
        for model_info in gemini_client.models.list():
            models.append(
                {
                    "name": model_info.name,
                    "display_name": (
                        model_info.display_name
                        if hasattr(model_info, "display_name")
                        else None
                    ),
                    "description": (
                        model_info.description
                        if hasattr(model_info, "description")
                        else None
                    ),
                    "is_tuned": "tuned" in model_info.name
                    or "-tuned" in model_info.name,
                }
            )

        # Sort models - tuned models first, then alphabetically
        models.sort(key=lambda x: (not x["is_tuned"], x["name"]))

        logger.info(f"Found {len(models)} models, including tuned models")
        return models

    except Exception as e:
        logger.exception(f"Error listing models: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list models: {str(e)}")
