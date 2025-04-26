from pydantic import BaseModel


class LLMQuery(BaseModel):
    user_input: str
    top_n_candidates: int = 5  # How many candidates to feed to the LLM


class RecommendationQuery(BaseModel):
    prompt: str
    top_n: int = 10  # How many final recommendations
