#!/usr/bin/env python3
"""
Movie Recommendation Model Tuning Script

This script uses the REST API directly to tune a model on the MovieLens dataset
and provides progress tracking during the tuning process.
"""

import os
import json
import time
import requests
import argparse
import pandas as pd
import numpy as np
from datetime import datetime

import config
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable is not set")

# API endpoints
BASE_URL = "https://generativelanguage.googleapis.com"
TUNED_MODELS_ENDPOINT = f"{BASE_URL}/v1beta/tunedModels"
OPERATIONS_ENDPOINT = f"{BASE_URL}/v1"


def create_tuning_dataset():
    """Creates a tuning dataset from the MovieLens small dataset.

    Returns:
        list: A list of examples in the format expected by the API
    """
    print("Creating tuning dataset from MovieLens dataset...")

    # Load the movies.csv file
    movies_df = pd.read_csv(config.MOVIES_CSV)

    # Load the ratings.csv file
    ratings_df = pd.read_csv(config.RATINGS_CSV)

    # Load the tags.csv file
    tags_df = pd.read_csv(config.TAGS_CSV)

    # Aggregate ratings by movieId to get average rating
    movie_ratings = (
        ratings_df.groupby("movieId")["rating"].agg(["mean", "count"]).reset_index()
    )
    movie_ratings = movie_ratings[
        movie_ratings["count"] > 10
    ]  # Only use movies with more than 10 ratings

    # Aggregate tags by movieId
    tags_agg = (
        tags_df.groupby("movieId")["tag"]
        .apply(lambda x: "|".join(x.unique()))
        .reset_index()
    )

    # Merge with movies data
    tuning_data = pd.merge(movies_df, movie_ratings, on="movieId")
    tuning_data = pd.merge(tuning_data, tags_agg, on="movieId", how="left")
    tuning_data["tag"].fillna(
        "popular", inplace=True
    )  # Default tag for movies without tags

    # Order by rating (highest first)
    tuning_data = tuning_data.sort_values(by="mean", ascending=False)

    # Extract year from title and create decade column
    tuning_data["year"] = tuning_data["title"].str.extract(r"\((\d{4})\)").astype(float)
    tuning_data["decade"] = (tuning_data["year"] // 10 * 10).astype(int)

    # Create input-output pairs for tuning
    tuning_examples = []

    # Create examples for genre-based recommendations
    genres = tuning_data["genres"].str.split("|").explode().unique()
    for genre in genres:
        # Find top rated movies for each genre
        genre_movies = tuning_data[tuning_data["genres"].str.contains(genre)]
        if len(genre_movies) >= 5:
            top_movies = genre_movies.head(5)

            input_text = f"Recommend me {genre} movies"

            # Structure output to match our expected format in recommendations
            output_lines = []
            for _, movie in top_movies.iterrows():
                output_lines.append(movie["title"])
                output_lines.append(movie["genres"])
                output_lines.append(movie["tag"])
                output_lines.append("")  # Empty line between entries

            output_text = "\n".join(output_lines).strip()

            # Ensure we're within the character limits for fine-tuning
            if len(input_text) <= 40000 and len(output_text) <= 5000:
                tuning_examples.append(
                    {"text_input": input_text, "output": output_text}
                )

    # Create examples for decade-based recommendations
    for decade in sorted(tuning_data["decade"].unique()):
        if not np.isnan(decade):
            decade_movies = tuning_data[tuning_data["decade"] == decade]
            if len(decade_movies) >= 5:
                top_movies = decade_movies.head(5)

                input_text = f"Recommend movies from the {int(decade)}s"

                # Structure output
                output_lines = []
                for _, movie in top_movies.iterrows():
                    output_lines.append(movie["title"])
                    output_lines.append(movie["genres"])
                    output_lines.append(movie["tag"])
                    output_lines.append("")  # Empty line between entries

                output_text = "\n".join(output_lines).strip()

                if len(input_text) <= 40000 and len(output_text) <= 5000:
                    tuning_examples.append(
                        {"text_input": input_text, "output": output_text}
                    )

    # Create examples for combined genre and decade queries
    for genre in genres[
        :10
    ]:  # Limit to first 10 genres to keep dataset size reasonable
        for decade in sorted(tuning_data["decade"].unique())[-4:]:  # Last 4 decades
            if not np.isnan(decade):
                genre_decade_movies = tuning_data[
                    (tuning_data["genres"].str.contains(genre))
                    & (tuning_data["decade"] == decade)
                ]
                if len(genre_decade_movies) >= 3:
                    top_movies = genre_decade_movies.head(3)

                    input_text = f"Recommend {genre} movies from the {int(decade)}s"

                    # Structure output
                    output_lines = []
                    for _, movie in top_movies.iterrows():
                        output_lines.append(movie["title"])
                        output_lines.append(movie["genres"])
                        output_lines.append(movie["tag"])
                        output_lines.append("")  # Empty line between entries

                    output_text = "\n".join(output_lines).strip()

                    if len(input_text) <= 40000 and len(output_text) <= 5000:
                        tuning_examples.append(
                            {"text_input": input_text, "output": output_text}
                        )

    print(f"Created {len(tuning_examples)} tuning examples")
    return tuning_examples


def start_model_tuning(model_name=None):
    """Start the model tuning process

    Args:
        model_name (str, optional): Display name for the model. Defaults to None.

    Returns:
        dict: The response from the tuning API
    """
    # Create a timestamp-based model name if none provided
    if model_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = f"movie_recommendation_model_{timestamp}"

    # Create the tuning dataset
    tuning_examples = create_tuning_dataset()

    # Prepare the API request
    url = f"{TUNED_MODELS_ENDPOINT}?key={GEMINI_API_KEY}"
    headers = {"Content-Type": "application/json"}

    # Build the request payload
    payload = {
        "display_name": model_name,
        "base_model": "models/gemini-1.5-flash-001-tuning",
        "tuning_task": {
            "hyperparameters": {
                "batch_size": 4,
                "learning_rate": 0.001,
                "epoch_count": 5,
            },
            "training_data": {"examples": {"examples": tuning_examples}},
        },
    }

    print(f"Starting model tuning for '{model_name}'...")
    response = requests.post(url, headers=headers, json=payload)

    if response.status_code == 200:
        result = response.json()

        # Save the response to a file
        with open("tunemodel.json", "w") as f:
            json.dump(result, f, indent=2)

        print(f"Tuning job started successfully. Operation: {result.get('name')}")
        return result
    else:
        print(f"Error starting tuning job: {response.status_code}")
        print(response.text)
        return None


def check_tuning_progress(operation_name):
    """Check the progress of a tuning operation

    Args:
        operation_name (str): The name of the operation returned by the tuning API

    Returns:
        tuple: (completion_percentage, is_done, tuned_model)
    """
    url = f"{OPERATIONS_ENDPOINT}/{operation_name}?key={GEMINI_API_KEY}"
    headers = {"Content-Type": "application/json"}

    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        result = response.json()

        # Save the latest operation status
        with open("tuning_operation.json", "w") as f:
            json.dump(result, f, indent=2)

        completed_percentage = result.get("metadata", {}).get("completedPercent", 0)
        is_done = result.get("done", False)
        tuned_model = result.get("metadata", {}).get("tunedModel", "")

        return (completed_percentage, is_done, tuned_model)
    else:
        print(f"Error checking tuning progress: {response.status_code}")
        print(response.text)
        return (0, False, "")


def check_tuned_model_status(model_name):
    """Check if a tuned model is ready to use

    Args:
        model_name (str): The name of the tuned model

    Returns:
        str: The state of the model
    """
    url = f"{TUNED_MODELS_ENDPOINT}/{model_name.split('/')[-1]}?key={GEMINI_API_KEY}"
    headers = {"Content-Type": "application/json"}

    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        result = response.json()

        # Save the model info
        with open("tuned_model.json", "w") as f:
            json.dump(result, f, indent=2)

        state = result.get("state", "UNKNOWN")
        return state
    else:
        print(f"Error checking model status: {response.status_code}")
        print(response.text)
        return "ERROR"


def monitor_tuning_progress(operation_name):
    """Monitor the tuning progress until completion

    Args:
        operation_name (str): The operation name returned by the tuning API
    """
    tuning_done = False
    tuned_model = ""

    print("Monitoring tuning progress...")

    while not tuning_done:
        completed_percentage, tuning_done, tuned_model = check_tuning_progress(
            operation_name
        )
        print(f"Tuning... {completed_percentage}%", end="\r")
        time.sleep(20)  # Check every 20 seconds

    print("\nTuning completed!")

    if tuned_model:
        print(f"Tuned model: {tuned_model}")
        state = check_tuned_model_status(tuned_model)
        print(f"Model state: {state}")

        # Write tuned model name to a file for easy access later
        with open("tuned_model_name.txt", "w") as f:
            f.write(tuned_model)

        print("Model name saved to tuned_model_name.txt")

        return tuned_model
    else:
        print("No tuned model was created.")
        return None


def main():
    """Main function to run the tuning process"""
    parser = argparse.ArgumentParser(description="Tune a movie recommendation model")
    parser.add_argument(
        "--model-name", type=str, help="Display name for the tuned model"
    )
    parser.add_argument(
        "--monitor-only",
        type=str,
        help="Only monitor an existing operation (provide operation name)",
    )
    args = parser.parse_args()

    if args.monitor_only:
        print(f"Monitoring existing operation: {args.monitor_only}")
        monitor_tuning_progress(args.monitor_only)
    else:
        # Start model tuning
        result = start_model_tuning(args.model_name)

        if result and "name" in result:
            operation_name = result["name"]
            monitor_tuning_progress(operation_name)
        else:
            print("Failed to start model tuning.")


if __name__ == "__main__":
    main()
