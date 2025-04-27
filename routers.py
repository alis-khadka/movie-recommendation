from fastapi import APIRouter, HTTPException

from schemas import LLMQuery, RecommendationQuery

from controllers import (
    generate_llm_recommendations,
    handle_recommendation_request,
    tune_movie_recommendation_model,
    generate_tuned_recommendations,
    list_available_models,
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


@router.post(
    "/tune_model",
    summary="Create a tuned model using MovieLens dataset",
    tags=["Tuned Model"],
)
async def tune_model():
    """
    Creates and tunes a model for movie recommendations using the MovieLens ml-latest-small dataset.
    Returns information about the tuning job.
    """
    try:
        tuning_job = tune_movie_recommendation_model()
        return {
            "message": "Model tuning job started successfully",
            "job_id": tuning_job.name,
            "model_name": (
                tuning_job.tuned_model.model
                if tuning_job.tuned_model
                else "Not yet available"
            ),
            "status": "In progress",
        }
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        print(f"Error in /tune_model endpoint: {e}")
        raise HTTPException(
            status_code=500, detail=f"Error creating model tuning job: {str(e)}"
        )


@router.post(
    "/recommend_tuned",
    summary="Get recommendations using tuned model",
    tags=["Tuned Model"],
)
async def recommend_tuned(query: RecommendationQuery, model_name: str):
    """
    Generates movie recommendations using a model that has been tuned on the MovieLens dataset.

    - query: The recommendation query with the user prompt
    - model_name: The name/ID of the tuned model to use (provided by the /tune_model endpoint)
    """
    try:
        return generate_tuned_recommendations(query, model_name)
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        print(f"Error in /recommend_tuned endpoint: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error generating recommendations with tuned model: {str(e)}",
        )


@router.get(
    "/models",
    summary="List all available models including tuned models",
    tags=["Tuned Model"],
)
async def get_models():
    """
    Lists all available models that can be used for recommendations,
    including any tuned models that have been created.
    """
    try:
        return list_available_models()
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        print(f"Error in /models endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Error listing models: {str(e)}")


@router.post(
    "/tune_model_with_progress",
    summary="Create a tuned model with progress tracking",
    tags=["Tuned Model"],
)
async def tune_model_with_progress(model_name: str = None):
    """
    Creates and tunes a model for movie recommendations using the MovieLens dataset,
    with progress tracking via direct REST API calls.

    This endpoint starts the tuning process and returns immediately with information
    about the tuning job. The actual tuning will continue in the background.

    Args:
        model_name (str, optional): Custom name for the tuned model
    """
    try:
        # Import the functions from our tune_model.py script
        from tune_model import start_model_tuning

        # Start the tuning process
        result = start_model_tuning(model_name)

        if result and "name" in result:
            return {
                "message": "Model tuning job started successfully",
                "operation_name": result["name"],
                "tuned_model": result.get("metadata", {}).get(
                    "tunedModel", "Not yet available"
                ),
                "status": "In progress",
                "monitor_command": f"python tune_model.py --monitor-only {result['name']}",
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to start model tuning")
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        print(f"Error in /tune_model_with_progress endpoint: {e}")
        raise HTTPException(
            status_code=500, detail=f"Error creating model tuning job: {str(e)}"
        )


@router.get(
    "/tuning_status/{operation_name}",
    summary="Get status of a tuning operation",
    tags=["Tuned Model"],
)
async def tuning_status(operation_name: str):
    """
    Check the status of a model tuning operation.

    Args:
        operation_name (str): The operation name returned by the tuning API
    """
    try:
        # Import the function from our tune_model.py script
        from tune_model import check_tuning_progress

        completed_percentage, is_done, tuned_model = check_tuning_progress(
            operation_name
        )

        return {
            "operation_name": operation_name,
            "completed_percentage": completed_percentage,
            "is_done": is_done,
            "tuned_model": tuned_model,
        }
    except Exception as e:
        print(f"Error in /tuning_status endpoint: {e}")
        raise HTTPException(
            status_code=500, detail=f"Error checking tuning status: {str(e)}"
        )
