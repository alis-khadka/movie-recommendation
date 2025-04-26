from pydantic import BaseModel


class LLMQuery(BaseModel):
    user_input: str
    top_n: int = 10  # How many candidates to retrieve for the LLM


class RecommendationQuery(BaseModel):
    prompt: str
    top_n: int = 10  # How many final recommendations
