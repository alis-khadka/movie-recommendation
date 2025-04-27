import requests
import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
import os

# API endpoints to test
ENDPOINTS = {
    "keyword": "http://127.0.0.1:8000/recommend",
    "llm": "http://127.0.0.1:8000/recommend_llm",
}

# List of prompts to test
PROMPTS = [
    "Suggest me mystery thriller movies from 1990s",
    "Suggest me movies starring Anthony Perkins",
    "The user wants to watch sci-fi movies from 2015",
    "romantic movies from 2004",
    "movies that have Salvador in the title",
    "movie related to war from 70s",
    "movie with crime comedy drama and thriller from 90s",
    "all Slumber Party Massacre movies",
    "Suggest me psychological thriller movies from the 1980s",
    "Japanese horror films released between 2000-2010",
    "Movies directed by Alfred Hitchcock in color",
    "Films featuring Meryl Streep as the main character",
    "Comedy movies set in high schools from the 1990s",
    "Action movies with female protagonists from 2010-2020",
    "French New Wave films from the 1960s",
    "Movies about time travel released after 2010",
    "Films that won both Best Picture and Best Director Oscars",
    "Space exploration documentaries from the last decade",
    "Movies set during the Great Depression",
    "Australian films with indigenous themes",
    "Cyberpunk movies released before Blade Runner",
    "Movies with twist endings from the 2000s",
    "Films shot entirely in one location",
    "Animated movies not made by Disney or Pixar",
    "Movies about writers struggling with creativity",
    "Korean thrillers from the last five years",
    "Historical dramas set in ancient Rome",
    "Movies featuring AI or robots as main characters",
    "Film noir classics from the 1940s",
    "Disaster movies from the 1970s",
    "Films based on Stephen King short stories",
    "British comedy movies from the 1960s",
    "Movies set during Christmas but aren't typical holiday films",
    "Films directed by women before 1980",
    "Boxing movies from any decade",
    "Movies with unreliable narrators",
    "Concert documentaries from the 1980s and 1990s",
    "Films set entirely on trains or submarines",
]


def measure_endpoint_latency(endpoint_name, endpoint_url, prompt, top_n=10):
    """Measure the response time of an API endpoint"""
    if endpoint_name == "keyword":
        payload = {"prompt": prompt, "top_n": top_n}
    else:  # llm endpoint
        payload = {"user_input": prompt, "top_n": top_n}

    start_time = time.time()
    try:
        response = requests.post(endpoint_url, json=payload)
        response.raise_for_status()  # Raise exception for HTTP errors
        response_time = time.time() - start_time
        return {
            "endpoint": endpoint_name,
            "prompt": prompt,
            "latency": response_time,
            "status_code": response.status_code,
            "success": True,
        }
    except Exception as e:
        response_time = time.time() - start_time
        # Handle various exception types more gracefully
        status_code = None
        if hasattr(e, "response") and e.response is not None:
            status_code = e.response.status_code

        error_message = str(e)
        # Special handling for connection errors
        if isinstance(e, requests.exceptions.ConnectionError):
            error_message = "Connection refused - server may not be running"

        return {
            "endpoint": endpoint_name,
            "prompt": prompt,
            "latency": response_time,
            "status_code": status_code,
            "success": False,
            "error": error_message,
        }


def run_benchmark():
    """Run benchmarking on all prompts and endpoints"""
    results = []

    for prompt in PROMPTS:
        print(f"Testing prompt: {prompt[:50]}...")

        for endpoint_name, endpoint_url in ENDPOINTS.items():
            print(f"  Testing endpoint: {endpoint_name}")
            result = measure_endpoint_latency(endpoint_name, endpoint_url, prompt)
            results.append(result)

            # Add a longer delay after LLM requests to avoid rate limiting
            if endpoint_name == "llm":
                print(f"    Adding 3-second delay after LLM request...")
                time.sleep(3.0)
            else:
                # Add a small delay to avoid overwhelming the server
                time.sleep(0.5)

    return results


def save_results(results):
    """Save benchmark results to CSV and JSON"""
    # Create results directory if it doesn't exist
    os.makedirs("benchmark_results", exist_ok=True)

    # Generate timestamp for filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save as CSV
    df = pd.DataFrame(results)
    csv_path = f"benchmark_results/api_latency_{timestamp}.csv"
    df.to_csv(csv_path, index=False)

    # Save as JSON
    json_path = f"benchmark_results/api_latency_{timestamp}.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    return df, csv_path, json_path


def create_visualization(df):
    """Create and save visualizations of the benchmark results"""
    # Create results directory if it doesn't exist
    os.makedirs("benchmark_results", exist_ok=True)

    # Generate timestamp for filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Filter out failed requests
    successful_df = df[df["success"] == True].copy()

    if successful_df.empty:
        print("No successful requests to visualize")
        return

    # Set the style
    sns.set(style="whitegrid")
    plt.figure(figsize=(12, 8))

    # Create a boxplot comparing the two endpoints
    ax = sns.boxplot(x="endpoint", y="latency", data=successful_df)
    plt.title("API Response Latency Comparison", fontsize=16)
    plt.xlabel("API Endpoint", fontsize=14)
    plt.ylabel("Response Time (seconds)", fontsize=14)

    # Save boxplot
    boxplot_path = f"benchmark_results/latency_boxplot_{timestamp}.png"
    plt.savefig(boxplot_path, dpi=300, bbox_inches="tight")
    plt.close()

    # Create a violin plot for more detailed distribution
    plt.figure(figsize=(12, 8))
    ax = sns.violinplot(x="endpoint", y="latency", data=successful_df)
    plt.title("API Response Latency Distribution", fontsize=16)
    plt.xlabel("API Endpoint", fontsize=14)
    plt.ylabel("Response Time (seconds)", fontsize=14)

    # Save violin plot
    violin_path = f"benchmark_results/latency_violin_{timestamp}.png"
    plt.savefig(violin_path, dpi=300, bbox_inches="tight")
    plt.close()

    # Create a line graph comparing latency between endpoints
    plt.figure(figsize=(14, 8))

    # Prepare data for line graph
    # Reset index to get prompt as a column and sort by prompt number
    plot_df = successful_df.copy()
    plot_df["prompt_number"] = plot_df.groupby("prompt").ngroup()

    # Create separate dataframes for each endpoint
    keyword_df = plot_df[plot_df["endpoint"] == "keyword"].sort_values("prompt_number")
    llm_df = plot_df[plot_df["endpoint"] == "llm"].sort_values("prompt_number")

    # Plot both lines
    plt.plot(
        keyword_df["prompt_number"],
        keyword_df["latency"],
        marker="o",
        linestyle="-",
        linewidth=2,
        markersize=6,
        label="keyword",
    )
    plt.plot(
        llm_df["prompt_number"],
        llm_df["latency"],
        marker="s",
        linestyle="-",
        linewidth=2,
        markersize=6,
        label="llm",
    )

    plt.title("API Latency Comparison by Prompt", fontsize=16)
    plt.xlabel("Prompt Number", fontsize=14)
    plt.ylabel("Response Time (seconds)", fontsize=14)
    plt.legend(title="Endpoint")
    plt.grid(True)
    plt.tight_layout()

    # Save line graph
    line_path = f"benchmark_results/latency_line_comparison_{timestamp}.png"
    plt.savefig(line_path, dpi=300, bbox_inches="tight")
    plt.close()

    # Create a bar chart showing average latency per prompt for each endpoint
    plt.figure(figsize=(15, 10))

    # Prepare data for grouped bar chart
    pivot_df = successful_df.pivot(index="prompt", columns="endpoint", values="latency")
    pivot_df = pivot_df.sort_values(by="keyword")  # Sort by keyword latency

    # Use only first few words of each prompt for readability
    short_prompts = [p[:20] + "..." for p in pivot_df.index]

    # Create the bar chart
    ax = pivot_df.plot(kind="bar", figsize=(15, 10))
    plt.title("Average Latency by Prompt and Endpoint", fontsize=16)
    plt.xlabel("Prompt", fontsize=14)
    plt.ylabel("Response Time (seconds)", fontsize=14)
    plt.xticks(range(len(short_prompts)), short_prompts, rotation=45, ha="right")
    plt.tight_layout()
    plt.legend(title="Endpoint")

    # Save bar chart
    bar_path = f"benchmark_results/latency_by_prompt_{timestamp}.png"
    plt.savefig(bar_path, dpi=300, bbox_inches="tight")
    plt.close()

    # Create a summary bar chart with mean and std dev
    plt.figure(figsize=(10, 6))
    summary = successful_df.groupby("endpoint")["latency"].agg(["mean", "std"])
    summary.plot(kind="bar", y="mean", yerr="std", capsize=10, figsize=(10, 6))
    plt.title("Mean API Response Latency with Standard Deviation", fontsize=16)
    plt.xlabel("API Endpoint", fontsize=14)
    plt.ylabel("Response Time (seconds)", fontsize=14)
    plt.tight_layout()

    # Save summary chart
    summary_path = f"benchmark_results/latency_summary_{timestamp}.png"
    plt.savefig(summary_path, dpi=300, bbox_inches="tight")
    plt.close()

    return boxplot_path, violin_path, line_path, bar_path, summary_path


def print_summary(df):
    """Print summary statistics of the benchmark results"""
    # Filter out failed requests
    successful_df = df[df["success"] == True]

    if successful_df.empty:
        print("No successful requests to summarize")
        return

    # Calculate summary statistics by endpoint
    summary = successful_df.groupby("endpoint")["latency"].agg(
        ["mean", "median", "min", "max", "std", "count"]
    )

    print("\n===== BENCHMARK SUMMARY =====")
    print(summary)
    print("\n")

    # Count failed requests
    failed = df[df["success"] == False]
    if not failed.empty:
        print(f"Failed requests: {len(failed)}")
        for _, row in failed.iterrows():
            print(
                f"  - Endpoint: {row['endpoint']}, Prompt: '{row['prompt'][:30]}...', Error: {row.get('error', 'Unknown')}"
            )
    else:
        print("All requests were successful!")


if __name__ == "__main__":
    print("Starting API latency benchmark...")
    results = run_benchmark()
    df, csv_path, json_path = save_results(results)
    print(f"Results saved to {csv_path} and {json_path}")

    try:
        visualization_paths = create_visualization(df)
        if visualization_paths:
            print(f"Visualizations saved to benchmark_results directory")
    except Exception as e:
        print(f"Error creating visualizations: {e}")

    print_summary(df)
