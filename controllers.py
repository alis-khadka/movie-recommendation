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
            # Use query.top_n for logging
            f"Processing input: {query.user_input}, top_n: {query.top_n}"
        )

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
        response = gemini_client.models.generate_content(
            model="models/gemini-2.0-flash",
            contents=prompt,
            config={
                "temperature": 0,
                "system_instruction": "You are a helpful movie assistant. Your response should not include any extra information, explanation or pretext.",
            },
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
            "raw_recommendations": recommendation_text,  # Renamed for clarity
            "parsed_recommendations": parsed_recommendations,
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
