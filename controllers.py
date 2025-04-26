import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import traceback
from fastapi import HTTPException
import logging

from schemas import LLMQuery, RecommendationQuery
from data_processing import movies_llm_df, embeddings_matrix
from models import gemini_client, embed_model
from recommenders import get_keyword_recommendations

logger = logging.getLogger(__name__)


def generate_llm_recommendations(query: LLMQuery):
    """Handles the logic for generating LLM-based recommendations."""
    try:
        logger.info(f"--- Inside LLM Controller Logic ---")
        logger.info(
            f"Processing input: {query.user_input}, top_n_candidates: {query.top_n_candidates}"
        )

        query_vec = embed_model.encode(query.user_input)
        similarities = cosine_similarity([query_vec], embeddings_matrix)[0]
        num_candidates = min(query.top_n_candidates, len(movies_llm_df))
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
You are a helpful movie assistant. Recommend only from the list below based on the user's preferences.

User Input: "{query.user_input}"

Available Movies:
{movie_list}

Please recommend the 2–3 most relevant movies and explain why.
"""
        logger.info("Sending prompt to Gemini...")
        # Updated Gemini API call and model name to match original snippet
        response = gemini_client.models.generate_content(
            model="models/gemini-1.5-flash-latest",  # Using the model from the original snippet
            contents=prompt,
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

        candidates_output = [
            {
                "title": row["title"],
                "genres": (
                    list(row["genres"]) if isinstance(row["genres"], list) else []
                ),
                "tags": list(row["tag"]) if isinstance(row["tag"], list) else [],
            }
            for _, row in top_movies.iterrows()
        ]

        return {
            "user_input": query.user_input,
            "recommendations": recommendation_text,
            "candidates": candidates_output,
        }

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
        logger.info(f"Processing input: {query.prompt}, top_n: {query.top_n}")
        recommendations_df = get_keyword_recommendations(query.prompt, query.top_n)
        return recommendations_df.to_dict(orient="records")
    except Exception as e:
        logger.exception(f"Error in Keyword controller: {e}")
        traceback.print_exc()
        # Re-raise as HTTPException for the router to catch
        raise HTTPException(
            status_code=500,
            detail=f"An internal error occurred in Keyword logic: {str(e)}",
        )
