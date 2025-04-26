# Movie Recommendation API

A sophisticated movie recommendation system built with FastAPI that offers two recommendation approaches:
1. Keyword/TF-IDF based recommendations using traditional NLP techniques
2. LLM-based recommendations using Google's Gemini API

## Features

- **Keyword-Based Recommendations**: Uses NLP techniques including TF-IDF vectorization, keyword extraction, and Bayesian average ratings
- **LLM-Based Recommendations**: Leverages Google's Gemini API for context-aware movie suggestions
- **REST API**: Built with FastAPI for high performance and easy integration
- **Hybrid Scoring**: Combines similarity metrics with rating information for better recommendations
- **Memory Efficient**: Uses lazy loading for large data structures like embeddings

## Getting Started

### Prerequisites

- Python 3.10+
- MovieLens dataset (small version included in the repository)
- Google Gemini API key

### Installation

1. Clone the repository
```bash
git clone <repository-url>
cd movie-recommendation
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Set up your Google Gemini API key in a `.env` file:
```
GEMINI_API_KEY=your_api_key_here
```

You can obtain a Gemini API key from the [Google AI Studio](https://ai.google.dev/gemini-api/docs/quickstart?lang=python)

### Running the API

Start the server with:
```bash
uvicorn main:app --reload
```

Visit the API documentation at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

## API Endpoints

### 1. Keyword-Based Recommendations
```
POST /recommend
```

Example request:
```json
{
  "prompt": "I want a funny action movie with car chases",
  "top_n": 5
}
```

### 2. LLM-Based Recommendations
```
POST /recommend_llm
```

Example request:
```json
{
  "user_input": "Looking for psychological thrillers with plot twists like The Sixth Sense",
  "top_n": 5
}
```

## Dataset

This project uses the MovieLens dataset (ml-latest-small) which contains:
- 100,000+ ratings
- 9,000+ movies
- 3,600+ tag applications

## Technical Implementation

- **Data Processing**: Uses pandas for efficient data manipulation and preprocessing
- **Text Processing**: Employs spacy, keybert, and scikit-learn for NLP tasks
- **Vector Embeddings**: Uses sentence-transformers for text embedding
- **LLM Integration**: Connects with Google Gemini for natural language understanding
- **Performance Optimization**: Implements lazy loading for large data structures

## License

This project is licensed under the terms of the included LICENSE file.