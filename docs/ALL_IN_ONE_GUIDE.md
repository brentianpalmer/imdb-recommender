# All-in-One Four-Stage IMDb Recommender

## Overview

The All-in-One Four-Stage IMDb Recommender is a sophisticated recommendation system that leverages machine learning techniques to provide personalized movie and TV show recommendations based on your IMDb ratings and watchlist.

Unlike simple popularity-based or collaborative filtering systems, this recommender implements a four-stage approach that models both exposure bias and personal preferences while optimizing for diversity.

## Four-Stage Architecture

### Stage 1: Feature Engineering 🔧

Constructs comprehensive feature vectors for each movie/show including:

- **Content Features**: Genres (TF-IDF encoded), decade bins, missing value indicators
- **Popularity Features**: IMDb rating (normalized), log-transformed vote counts  
- **Temporal Features**: Recency with exponential decay (λ ≈ 0.03)
- **Runtime Features**: Duration bins (when available)

### Stage 2: Exposure Modeling 📺

Models P(exposed) - the probability that you've been exposed to a particular title. This addresses selection bias since you can only rate movies you've seen.

Since our catalog contains only rated/watchlisted items, we use a heuristic approach:
```
Exposure = 0.4 × IMDb_rating + 0.4 × log(votes) + 0.2 × recency_decay
```

### Stage 3: Preference Modeling ❤️

Learns P(like|exposed) using pairwise preference learning:
- Constructs training pairs: A ≻ B if rating(A) >= rating(B) + 2
- Uses feature differences for robust preference learning
- Handles cases with insufficient preference data gracefully

The personal score combines both models:
```
Personal Score = P(exposed) × P(like|exposed)
```

### Stage 4: Diversity Optimization 🎲

Applies Maximal Marginal Relevance (MMR) re-ranking:
- Projects items into latent space using TruncatedSVD (k=64)
- Balances relevance with diversity (λ=0.8)
- Prevents over-specialization in recommendations

## Final Scoring

The final recommendation score blends personal and popularity components using z-score normalization to prevent any single component from dominating:

```python
Personal_z = (Personal - μ_personal) / (σ_personal + 1e-8)
Popularity_z = (Popularity - μ_popularity) / (σ_popularity + 1e-8)  

Final Score = 0.7 × Personal_z + 0.3 × Popularity_z
```

This ensures both personal preferences and popularity signals contribute meaningfully to the final ranking, regardless of their original scale differences.

Where popularity is calculated as:

```python
Popularity = IMDb_rating × log(1+votes) × exp(-λ×age)
```

Then z-score normalized as described above.

## Usage

### Command Line Interface

Basic usage:
```bash
imdbrec all-in-one --config config.toml --topk 25
```

With evaluation and export:
```bash
imdbrec all-in-one --config config.toml --topk 50 --evaluate --export-csv recommendations.csv
```

Advanced options:
```bash
imdbrec all-in-one \
  --ratings data/ratings.csv \
  --watchlist data/watchlist.xlsx \
  --topk 25 \
  --user-weight 0.8 \
  --global-weight 0.2 \
  --save-model model.pkl \
  --export-csv recs.csv \
  --evaluate
```

### Parameters

- `--topk`: Number of recommendations to return (default: 25)
- `--user-weight`: Weight for personal preferences (default: 0.7)  
- `--global-weight`: Weight for popularity priors (default: 0.3)
- `--exclude-rated`: Exclude already rated items (default: True)
- `--save-model`: Path to save trained model
- `--export-csv`: Path to export recommendations CSV
- `--evaluate`: Run temporal split evaluation

### Python API

```python
from imdb_recommender.recommender_all_in_one import AllInOneRecommender
from imdb_recommender.data_io import ingest_sources

# Load data
res = ingest_sources("ratings.csv", "watchlist.xlsx")

# Initialize recommender  
recommender = AllInOneRecommender(res.dataset, random_seed=42)

# Generate recommendations
scores, explanations = recommender.score(
    seeds=[],
    user_weight=0.7,
    global_weight=0.3,
    exclude_rated=True
)

# Export recommendations
recommender.export_recommendations_csv(scores, "recommendations.csv", top_k=50)

# Evaluate performance
metrics = recommender.evaluate_temporal_split(test_size=0.2)
print(f"Hits@10: {metrics.get('hits_at_10', 0):.4f}")
print(f"NDCG@10: {metrics.get('ndcg_at_10', 0):.4f}")
print(f"Diversity: {metrics.get('diversity', 0):.4f}")

# Save/load trained model
recommender.save_model("model.pkl")
recommender.load_model("model.pkl")
```

## Output Formats

### Recommendations CSV

The exported CSV contains detailed recommendation information:

```csv
tconst,title,year,genres,title_type,imdb_rating,num_votes,runtime,score_personal,score_pop,score_final
tt7366338,Chernobyl,2019,"Drama, History, Thriller",movie,9.3,968181,Unknown,0.670,1.247,1.246
tt9362722,Spider-Man: Across the Spider-Verse,2023,"Animation, Action, Adventure",movie,8.5,474003,Unknown,0.665,1.168,1.168
```

### Explanation System

Each recommendation includes intelligent explanations:
- **"strong personal fit"**: High personal preference score (>0.6)
- **"moderate personal interest"**: Medium personal score (0.4-0.6)
- **"exploratory pick"**: Lower personal score (<0.4)
- **"high critical acclaim"**: High popularity score (>1.0)  
- **"well-regarded"**: Good popularity score (>0.5)
- **"likely in your sphere"**: High exposure probability (>0.7)
- **"hidden gem"**: Low exposure probability (<0.3)

## Evaluation Methodology

The recommender uses temporal split evaluation:

1. **Data Split**: Sort ratings by date, use earliest 80% for training, latest 20% for testing
2. **Metrics**:
   - **Hits@10**: Fraction of test items appearing in top-10 recommendations
   - **NDCG@10**: Normalized Discounted Cumulative Gain (rewards ranking test favorites higher)
   - **Diversity**: 1 - average pairwise cosine similarity in latent space (higher = more diverse)

### Baseline Comparisons

The system can be compared against:
- Popularity-only baseline
- Content similarity-only baseline  
- Simple 70/30 blend without exposure modeling

## Advanced Features

### Recommendation Shelves

Organizes recommendations into intuitive categories:

1. **Tonight Picks**: Short runtime (≤120min), recent (≥2016), high final score
2. **New & Aligned**: Recent releases (≥2018), high personal score
3. **Prestige Backlog**: Older films (<2010), high popularity score
4. **Stretch Picks**: High-scoring items with diversity from favorites

### Model Persistence

The trained model can be saved and loaded:
- Serializes all model components (exposure, preference, SVD, scalers)
- Preserves hyperparameters and feature encoders
- Enables fast inference without retraining

### Robustness Features

- Handles missing data gracefully (runtime, year, ratings)
- Works with small datasets (degrades to simpler models)
- Provides fallback explanations when models fail
- Validates input data and provides meaningful error messages

## Technical Implementation

### Dependencies

- `scikit-learn`: Machine learning models (LogisticRegression, TruncatedSVD)
- `pandas`: Data manipulation and feature engineering
- `numpy`: Numerical computations and array operations

### Performance Characteristics

- **Training Time**: O(n²) for pairwise preference learning, O(n) for other stages
- **Memory Usage**: O(n×k) where n=items, k=features (typically k≈50)
- **Inference Time**: O(n) for scoring, O(k²) for MMR re-ranking

### Hyperparameters

| Parameter           | Default | Description                        |
| ------------------- | ------- | ---------------------------------- |
| `recency_lambda`    | 0.03    | Exponential decay rate for recency |
| `personal_weight`   | 0.7     | Weight for personal preferences    |
| `popularity_weight` | 0.3     | Weight for popularity priors       |
| `mmr_lambda`        | 0.8     | MMR diversity parameter            |
| `svd_components`    | 64      | Latent space dimensionality        |

## Limitations and Future Enhancements

### Current Limitations

1. **Cold Start**: Requires existing ratings/watchlist data
2. **Runtime Data**: Not available in current IMDb exports
3. **Categorical Features**: Limited to genres (no cast, director, etc.)
4. **Preference Learning**: May struggle with very sparse rating data

### Potential Enhancements

1. **Content Expansion**: Add cast, director, plot keywords from IMDb API
2. **Deep Learning**: Replace linear models with neural networks
3. **Sequential Modeling**: Account for temporal rating patterns  
4. **Multi-Objective**: Optimize for multiple criteria (novelty, serendipity)
5. **Active Learning**: Suggest items to rate for better learning

## Research Background

This implementation draws from several research areas:

- **Recommendation Systems**: Collaborative filtering, content-based filtering
- **Bias Modeling**: Exposure bias, selection bias in implicit feedback
- **Learning to Rank**: Pairwise preference learning, ranking optimization
- **Diversity**: Maximal Marginal Relevance, portfolio optimization
- **Evaluation**: Temporal splitting, offline evaluation metrics

The four-stage architecture addresses common issues in recommendation systems while providing interpretable, actionable recommendations for personal movie discovery.

---

*For questions or issues, please refer to the main project documentation or open an issue on the project repository.*
