from fastapi import APIRouter, HTTPException

from schemas import LLMQuery, RecommendationQuery

from controllers import (
    generate_llm_recommendations,
    handle_recommendation_request,
)

router = APIRouter()


@router.post(
    "/recommend_llm", summary="Get recommendations via LLM", tags=["LLM Recommender"]
)
def recommend_movies_llm(query: LLMQuery):
    """
    Generates movie recommendations using a Large Language Model (Gemini)
    based on semantic similarity of the user input to movie descriptions.
    Delegates logic to the controller.
    """
    try:
        return generate_llm_recommendations(query)
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        print(f"Unexpected error in /recommend_llm router: {e}")
        raise HTTPException(
            status_code=500, detail=f"An unexpected internal error occurred: {str(e)}"
        )


@router.post(
    "/recommend",
    summary="Get recommendations via Keywords/TF-IDF",
    tags=["Keyword Recommender"],
)
async def recommend_keyword(query: RecommendationQuery):
    """
    Generates movie recommendations using keyword extraction, TF-IDF similarity on genres,
    and a hybrid scoring approach combining similarity and Bayesian average rating.
    Delegates logic to the controller.
    """
    try:
        return handle_recommendation_request(query)
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        print(f"Unexpected error in /recommend router: {e}")
        raise HTTPException(
            status_code=500, detail=f"An unexpected internal error occurred: {str(e)}"
        )


@router.get("/", summary="API Root", tags=["General"])
async def read_root():
    """Provides basic information about the API."""
    return {
        "message": "Welcome to the Movie Recommendation API. Use /docs for details."
    }
