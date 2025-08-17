# IMDb Recommender

A local movie recommendation system that analyzes your IMDb ratings and watchlist to generate personalized recommendations. The system combines collaborative filtering, content-based filtering, and popularity-based recommendations to suggest movies and TV shows you might enjoy.

## Overview

This project creates a sophisticated recommendation engine that:

- Ingests your IMDb ratings and watchlist data
- Uses multiple recommendation algorithms (Popularity-based and SVD matrix factorization)
- Blends recommendations using content similarity, user preferences, and global popularity
- Provides explainable recommendations with detailed reasoning
- Logs your rating and watchlist actions for future model improvements
- Supports Selenium replay for automated IMDb interactions

## Features

### Recommendation Algorithms

1. **Popularity-based Recommender** (`PopSimRecommender`): Uses content similarity and global popularity scores
2. **SVD Matrix Factorization** (`SVDAutoRecommender`): Employs collaborative filtering with latent factors
3. **Blended Approach**: Combines multiple algorithms for better recommendations

### Content Analysis

- **Genre Similarity**: Analyzes genre overlap with your highly-rated movies
- **Temporal Features**: Considers release year trends in your viewing preferences
- **Popularity Weighting**: Balances niche vs. mainstream recommendations
- **Recency Bias**: Optional weighting toward newer releases

### Explainable AI

Each recommendation comes with explanations like:

- "High genre overlap with your 8–10s; strong global acclaim"
- "Similar to your seed titles"
- "Latent-factor fit from your ratings and IMDb priors"

## Installation

### Prerequisites

- Python 3.10 or higher
- Virtual environment (recommended)

### Setup

```bash
# Clone the repository
cd /path/to/imdb_recommender_pkg

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -e .
pip install pyarrow scikit-learn toml  # Additional dependencies
```

## Data Requirements

### Input Files

1. **Ratings CSV** (`ratings.csv`): Your IMDb ratings export
   - Required columns: `Const`, `Your Rating`, `Title`, `Year`, `Genres`, `IMDb Rating`, `Num Votes`

2. **Watchlist Excel/CSV** (`watchlist.xlsx`): Your IMDb watchlist export
   - Required columns: `Const`, `Title`, `Year`, `Genres`, `IMDb Rating`, `Num Votes`

### Configuration

Create a `config.toml` file:

```toml
[data]
ratings_csv_path = "data/raw/ratings.csv"
watchlist_path = "data/raw/watchlist.xlsx"
movies_path = "data/processed/movies.parquet"
ratings_path = "data/processed/ratings.parquet"
watchlist_processed_path = "data/processed/watchlist.parquet"

[recommendation]
user_weight = 0.7
global_weight = 0.3
recency_weight = 0.5
topk = 25
```

## Usage

### Command Line Interface

The main CLI command is `imdbrec` with several subcommands:

#### 1. Data Ingestion

Process your IMDb exports:

```bash
imdbrec ingest --ratings data/raw/ratings.csv --watchlist data/raw/watchlist.xlsx
```

#### 2. Generate Recommendations

Get personalized movie recommendations:

```bash
# Using config file
imdbrec recommend --config config.toml --seeds tt0480249 --topk 25

# Direct parameters
imdbrec recommend --ratings data/raw/ratings.csv --watchlist data/raw/watchlist.xlsx --seeds tt0480249 --topk 25 --user-weight 0.7 --global-weight 0.3 --recency 0.5
```

**Parameters:**

- `--seeds`: IMDb ID(s) to use as recommendation seeds (comma-separated)
- `--topk`: Number of recommendations to return (default: 25)
- `--user-weight`: Weight for user preference similarity (default: 0.7)
- `--global-weight`: Weight for global popularity (default: 0.3)
- `--recency`: Bias toward newer releases (default: 0.0)
- `--exclude-rated`: Exclude already-rated items (default: true)

#### 3. Rate Movies

Log new ratings:

```bash
imdbrec rate tt1234567 8 --notes "Great cinematography"
```

#### 4. Manage Watchlist

Add or remove items from watchlist:

```bash
imdbrec watchlist add tt1234567
imdbrec watchlist remove tt1234567
```

#### 5. Quick Review

Rate and/or add to watchlist in one command:

```bash
imdbrec quick-review tt1234567 --rating 9 --watchlist add --notes "Must watch again"
```

#### 6. Export Logs

Export your rating/watchlist actions:

```bash
imdbrec export-log --out data/my_actions.csv
```

#### 7. Explain Recommendations

Get detailed explanation for why a specific movie was recommended:

```bash
imdbrec explain tt1234567 --ratings data/raw/ratings.csv --watchlist data/raw/watchlist.xlsx
```

## How It Works

### 1. Data Processing

The system ingests your IMDb data and:

- Normalizes ratings and metadata
- Extracts genre vectors and temporal features
- Builds a content catalog with similarity metrics
- Handles missing data (NaN values) gracefully

### 2. Feature Engineering

- **Genre Vectors**: One-hot encoded and normalized genre representations
- **Year Buckets**: Temporal features binned by decade (1980, 1990, 2000, 2010, 2020+)
- **Content Vectors**: Combined genre and temporal feature vectors
- **Popularity Scores**: Normalized IMDb ratings with vote count bonuses

### 3. Recommendation Generation

The system uses two main approaches:

#### Popularity-based Recommender

- Calculates content similarity between seed titles and candidates
- Weights by user preference patterns and global popularity
- Applies recency bias if specified
- Generates explanations based on similarity factors

#### SVD Matrix Factorization

- Builds user-item matrix from ratings
- Applies singular value decomposition with optimal component selection
- Handles cold-start problems with global priors
- Provides latent factor explanations

#### Blending Strategy

- Combines scores from multiple algorithms using weighted averaging
- Ranks final recommendations by blended score, popularity, and recency
- Filters out already-rated items if requested

### 4. Explainability

Each recommendation includes human-readable explanations:

- Content similarity measures ("high genre overlap")
- Popularity indicators ("strong global acclaim")  
- Collaborative filtering insights ("similar to your seed titles")
- Algorithm attribution ("latent-factor fit")

## Project Structure

```text
imdb_recommender/
├── __init__.py
├── cli.py                 # Command-line interface
├── config.py             # Configuration management
├── data_io.py            # Data ingestion and processing
├── features.py           # Feature engineering (genres, years, similarity)
├── logger.py             # Action logging for Selenium replay
├── ranker.py             # Recommendation ranking and blending
├── recommender_base.py   # Base recommender class
├── recommender_pop.py    # Popularity-based recommender
├── recommender_svd.py    # SVD matrix factorization recommender
├── schemas.py            # Data models and schemas
└── selenium_stub/        # Selenium automation utilities
    └── replay_from_csv.py
```

## Advanced Usage

### Custom Weighting

Adjust algorithm weights for different recommendation styles:

- High `user_weight` (0.8-0.9): More personalized, niche recommendations
- High `global_weight` (0.4-0.6): More mainstream, popular recommendations
- High `recency` (0.5-1.0): Bias toward newer releases

### Seed-based Recommendations

Use specific movies as seeds to get similar recommendations:

```bash
# Get movies similar to The Dark Knight
imdbrec recommend --seeds tt0468569 --config config.toml
```

### Multi-seed Recommendations

Combine multiple seeds for complex preferences:

```bash
# Get movies similar to both Inception and Interstellar  
imdbrec recommend --seeds "tt1375666,tt0816692" --config config.toml
```

## Troubleshooting

### Common Issues

1. **Command not found**: Use full path or activate virtual environment

   ```bash
   source .venv/bin/activate
   # or
   /path/to/.venv/bin/imdbrec --help
   ```

2. **Missing dependencies**: Install additional packages

   ```bash
   pip install pyarrow scikit-learn toml
   ```

3. **Data format errors**: Ensure your IMDb exports match expected column names

4. **NA/Missing values**: The system handles missing data automatically

### Virtual Environment

If `imdbrec` command is not found, activate your virtual environment:

```bash
source /Users/brent/workspace/imdb_recommender_pkg/.venv/bin/activate
imdbrec --help
```

## Contributing

This is a personal movie recommendation system. Contributions welcome for:

- Additional recommendation algorithms
- Better feature engineering
- UI improvements
- Performance optimizations

## License

MIT License - see LICENSE file for details.
