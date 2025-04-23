from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from google import genai
import os
from dotenv import load_dotenv

load_dotenv()

# Paths
MOVIES_CSV = "./ml-latest-small/movies.csv"
TAGS_CSV = "./ml-latest-small/tags.csv"
EMBEDDING_PATH = "movies_embeddings.npy"
METADATA_PATH = "movies_metadata.parquet"

# Load or compute embeddings
if os.path.exists(METADATA_PATH) and os.path.exists(EMBEDDING_PATH):
    movies = pd.read_parquet(METADATA_PATH)
    embeddings = np.load(EMBEDDING_PATH)
else:
    # Load datasets
    movies = pd.read_csv("./ml-latest-small/movies.csv")
    tags = pd.read_csv("./ml-latest-small/tags.csv")

    # Preprocess genres (pipe to list)
    movies["genres"] = movies["genres"].fillna("").apply(lambda g: g.split("|"))

    # Preprocess tags (group by movieId)
    tags_grouped = tags.groupby("movieId")["tag"].apply(lambda x: list(set(x))).reset_index()
    movies = movies.merge(tags_grouped, on="movieId", how="left")
    movies["tag"] = movies["tag"].apply(lambda t: t if isinstance(t, list) else [])

    # Combine text for embedding
    def build_text(row):
        title = row["title"]
        genres = ", ".join(row["genres"])
        tags = ", ".join(row["tag"])
        return f"{title}. Genres: {genres}. Tags: {tags}"

    movies["combined_text"] = movies.apply(build_text, axis=1)

    # Compute embeddings
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    movies["embedding"] = movies["combined_text"].apply(lambda x: embed_model.encode(x))

    embeddings = np.vstack(movies["embedding"].values)

    # Drop embedding column for saving metadata
    movies.drop(columns=["combined_text", "embedding"], inplace=True)
    movies.to_parquet(METADATA_PATH, index=False)
    np.save(EMBEDDING_PATH, embeddings)

# Gemini API Key
gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# FastAPI setup
app = FastAPI()

class Query(BaseModel):
    user_input: str

@app.post("/recommend_llm")
def recommend_movies(query: Query):
    try:
        # Embed user query
        embed_model = SentenceTransformer("all-MiniLM-L6-v2")
        query_vec = embed_model.encode(query.user_input)

        # Find top 5 matches
        similarities = cosine_similarity([query_vec], embeddings)[0]
        top_indices = np.argsort(similarities)[::-1][:5]
        top_movies = movies.iloc[top_indices]

        # Format movie context for Gemini
        movie_list = "\n".join(
            f"{i+1}. {row['title']} (Genres: {', '.join(row['genres'])}; Tags: {', '.join(row['tag'])})"
            for i, row in top_movies.iterrows()
        )

        # Gemini prompt
        prompt = f"""
You are a helpful movie assistant. Recommend only from the list below based on the user's preferences.

User Input: "{query.user_input}"

Available Movies:
{movie_list}

Please recommend the 2–3 most relevant movies and explain why.
"""

        response = gemini_client.models.generate_content(
            model="gemini-2.0-flash", contents=prompt
        )
        print(response.text)
        return {
            "user_input": query.user_input,
            "recommendations": response.text.strip(),
            "candidates": [
                {
                    "title": row["title"],
                    "genres": list(row["genres"]) if not isinstance(row["genres"], list) else row["genres"],
                    "tags": list(row["tag"]) if not isinstance(row["tag"], list) else row["tag"]
                }
                for _, row in top_movies.iterrows()
            ]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
