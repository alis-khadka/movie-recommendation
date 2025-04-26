import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import traceback
from fastapi import HTTPException
import logging
import time
from google.api_core import exceptions as api_exceptions

from schemas import LLMQuery, RecommendationQuery
from data_processing import movies_llm_df, get_embeddings_matrix
from models import gemini_client, embed_model
from recommenders import get_keyword_recommendations

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
            logger.info(
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
        logger.info(f"--- Inside Keyword Controller Logic ---")
        # Use 'prompt' field from RecommendationQuery schema instead of 'query'
        logger.info(f"Processing input: {query.prompt}, top_n: {query.top_n}")

        # Pass the user query to the keyword recommender
        recommendations_df = get_keyword_recommendations(query.prompt, query.top_n)

        return {
            "query": query.prompt,
            "recommendations": recommendations_df.to_dict(orient="records"),
        }
    except Exception as e:
        logger.exception(f"Error in keyword controller: {e}")
        traceback.print_exc()
        # Re-raise as HTTPException for the router to catch
        raise HTTPException(
            status_code=500,
            detail=f"An internal error occurred in keyword recommendation logic: {str(e)}",
        )
