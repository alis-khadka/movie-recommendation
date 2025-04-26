import os

from dotenv import load_dotenv


load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "ml-latest-small")
MOVIES_CSV = os.path.join(DATA_DIR, "movies.csv")
RATINGS_CSV = os.path.join(DATA_DIR, "ratings.csv")
TAGS_CSV = os.path.join(DATA_DIR, "tags.csv")
EMBEDDING_PATH = os.path.join(BASE_DIR, "movies_embeddings.npy")
METADATA_PATH = os.path.join(BASE_DIR, "movies_metadata.parquet")
SPACY_MODEL_NAME = "en_core_web_lg"

# Genre Mapping Dictionary
GENRE_MAPPING = {
    "comedy": ["funny", "humorous", "hilarious", "sitcom", "laugh"],
    "horror": ["scary", "creepy", "terrifying", "ghost", "zombie", "spooky", "gore"],
    "action": ["fight", "explosion", "combat", "chase", "battle", "gun", "stunt"],
    "romance": ["love", "relationship", "dating", "couple", "romantic"],
    "sci-fi": ["science fiction", "space", "alien", "future", "robot", "cyberpunk"],
    "thriller": ["suspense", "mystery", "tense", "intrigue"],
    "drama": ["serious", "story", "character", "emotional"],
    "fantasy": ["magic", "myth", "dragon", "wizard"],
    "animation": ["cartoon", "animated"],
    "documentary": ["real", "non-fiction", "history", "nature"],
}

# Recommendation Weights
SIMILARITY_WEIGHT = 0.7
RATING_WEIGHT = 0.3

print("Configuration loaded.")
