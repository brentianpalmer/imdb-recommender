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

1. **All-in-One Four-Stage Recommender** (`AllInOneRecommender`): Advanced ML-based system with:
   - **Exposure bias modeling**: Accounts for what you're likely to have seen
   - **Pairwise preference learning**: Learns from rating comparisons
   - **Candidate restriction**: Realistic recommendation pools instead of full catalog
   - **Z-score normalization**: Prevents popularity bias in final scoring
   - **MMR diversity optimization**: Ensures diverse recommendations
   - **Temporal split evaluation**: Robust model validation
   - **Content type filtering**: Movies, TV Series, Mini Series, etc.
   - **Watchlist-specific recommendations**: Focus on your saved items
   - **⭐ Recommended for most users**

2. **Popularity-based Recommender** (`PopSimRecommender`): Uses content similarity and global popularity scores

3. **SVD Matrix Factorization** (`SVDAutoRecommender`): Employs collaborative filtering with latent factors

4. **Blended Approach**: Combines multiple algorithms for better recommendations

### Smart Filtering & Personalization

- **Content Type Filtering**: Focus on Movies, TV Series, TV Mini Series, or any specific content type
- **Watchlist Recommendations**: Get recommendations specifically from your unrated watchlist items
- **Candidate Restriction**: Realistic recommendation pools (500-1000 items) instead of full catalog evaluation
- **Z-Score Normalization**: Balanced scoring that prevents popularity from dominating personal preferences
- **Exposure Modeling**: Accounts for what content you've likely been exposed to

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

The main CLI command is `python -m imdb_recommender.cli` with several subcommands:

#### 1. Data Ingestion

Process your IMDb exports:

```bash
python -m imdb_recommender.cli ingest --ratings data/raw/ratings.csv --watchlist data/raw/watchlist.xlsx
```

#### 2. All-in-One Four-Stage Recommender (⭐ Recommended)

The most advanced recommendation system with machine learning:

##### Basic Usage

```bash
# Basic recommendations using config
python -m imdb_recommender.cli all-in-one --config config.toml --topk 25

# Direct parameters
python -m imdb_recommender.cli all-in-one --ratings data/raw/ratings.csv --watchlist data/raw/watchlist.xlsx --topk 25
```

##### Content Type Filtering 🎬

Get recommendations for specific content types:

```bash
# Movies only
python -m imdb_recommender.cli all-in-one --config config.toml --content-type "Movie" --topk 10

# TV Series only
python -m imdb_recommender.cli all-in-one --config config.toml --content-type "TV Series" --topk 10

# TV Mini Series only
python -m imdb_recommender.cli all-in-one --config config.toml --content-type "TV Mini Series" --topk 5
```

Available content types: `Movie`, `TV Series`, `TV Mini Series`, `TV Special`, `TV Movie`, `Video`, `Short`, `TV Episode`, `Music Video`

##### Watchlist-Specific Recommendations 🎯

Get recommendations from your unrated watchlist items:

```bash
# All unrated watchlist items
python -m imdb_recommender.cli all-in-one --config config.toml --watchlist-only --topk 10

# Movies from your watchlist only
python -m imdb_recommender.cli all-in-one --config config.toml --watchlist-only --content-type "Movie" --topk 10

# TV Series from your watchlist only
python -m imdb_recommender.cli all-in-one --config config.toml --watchlist-only --content-type "TV Series" --topk 5
```

##### Advanced Options

```bash
# With evaluation and model export
python -m imdb_recommender.cli all-in-one --config config.toml --topk 50 --evaluate --export-csv recommendations.csv --save-model model.pkl

# Custom weights and candidate pool size
python -m imdb_recommender.cli all-in-one --config config.toml --user-weight 0.8 --global-weight 0.2 --candidates 1000
```

**Parameters:**

- `--topk`: Number of recommendations to return (default: 25)
- `--user-weight`: Weight for personal preferences (default: 0.7)
- `--global-weight`: Weight for popularity (default: 0.3)
- `--exclude-rated`: Exclude already-rated items (default: true)
- `--watchlist-only`: Only recommend from unrated watchlist items
- `--content-type`: Filter by content type (Movie, TV Series, etc.)
- `--candidates`: Size of candidate pool for non-watchlist mode (default: 500)
- `--save-model`: Path to save trained model
- `--export-csv`: Path to export recommendations CSV
- `--evaluate`: Run evaluation with temporal split

See [All-in-One Guide](docs/ALL_IN_ONE_GUIDE.md) for detailed documentation.

#### 3. Traditional Blended Recommender

Get personalized movie recommendations using the original system:

```bash
# Using config file
python -m imdb_recommender.cli recommend --config config.toml --seeds tt0480249 --topk 25

# Direct parameters
python -m imdb_recommender.cli recommend --ratings data/raw/ratings.csv --watchlist data/raw/watchlist.xlsx --seeds tt0480249 --topk 25 --user-weight 0.7 --global-weight 0.3 --recency 0.5
```

**Parameters:**

- `--seeds`: IMDb ID(s) to use as recommendation seeds (comma-separated)
- `--topk`: Number of recommendations to return (default: 25)
- `--user-weight`: Weight for user preference similarity (default: 0.7)
- `--global-weight`: Weight for global popularity (default: 0.3)
- `--recency`: Bias toward newer releases (default: 0.0)
- `--exclude-rated`: Exclude already-rated items (default: true)

#### 4. Rate Movies

Log new ratings:

```bash
python -m imdb_recommender.cli rate tt1234567 8 --notes "Great cinematography"
```

#### 5. Manage Watchlist

Add or remove items from watchlist:

```bash
python -m imdb_recommender.cli watchlist add tt1234567
python -m imdb_recommender.cli watchlist remove tt1234567
```

#### 6. Quick Review

Rate and/or add to watchlist in one command:

```bash
python -m imdb_recommender.cli quick-review tt1234567 --rating 9 --watchlist add --notes "Must watch again"
```

#### 7. Export Logs

Export your rating/watchlist actions:

```bash
python -m imdb_recommender.cli export-log --out data/my_actions.csv
```

#### 8. Explain Recommendations

Get detailed explanation for why a specific movie was recommended:

```bash
python -m imdb_recommender.cli explain tt1234567 --ratings data/raw/ratings.csv --watchlist data/raw/watchlist.xlsx
```

## Recent Improvements 🚀

### Advanced Filtering & Personalization

- **🎬 Content Type Filtering**: Filter recommendations by Movies, TV Series, TV Mini Series, and more
- **🎯 Watchlist-Specific Mode**: Get recommendations exclusively from your unrated watchlist items  
- **🧠 Intelligent Candidate Building**: Realistic recommendation pools instead of evaluating entire catalog
- **⚖️ Z-Score Normalization**: Balanced scoring prevents popularity bias from dominating personal preferences
- **🔄 Candidate Restriction**: Smart 500-1000 item pools combining watchlist, popular, and neighbor items

### Technical Enhancements

- **Exposure Bias Modeling**: Accounts for content you've likely been exposed to
- **Pairwise Preference Learning**: Learns from comparisons between your ratings  
- **Temporal Split Evaluation**: Robust validation using time-based train/test splits
- **Enhanced CLI**: Comprehensive command-line interface with helpful emoji indicators
- **Comprehensive Testing**: 46 automated tests ensuring reliability

### Example Workflows

```bash
# Get top movies from your watchlist you're most likely to rate highly
python -m imdb_recommender.cli all-in-one --watchlist-only --content-type "Movie" --topk 10 --config config.toml

# Find great TV series you haven't rated yet
python -m imdb_recommender.cli all-in-one --content-type "TV Series" --topk 15 --config config.toml

# Export your top movie recommendations to CSV for later
python -m imdb_recommender.cli all-in-one --content-type "Movie" --export-csv my_movie_recs.csv --config config.toml
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

### 3. All-in-One Four-Stage Recommendation Process

The advanced recommendation system follows a sophisticated four-stage pipeline:

#### Stage 1: Feature Engineering & Candidate Building
- **Intelligent Candidate Selection**: Builds realistic candidate pools combining:
  - Top unrated watchlist items (diverse content you've saved)  
  - Globally popular high-quality items (broad appeal)
  - Content-based neighbors (similar to your preferences)
- **Multi-dimensional Features**: Genre vectors, temporal features, popularity metrics
- **Content Type Filtering**: Focus on Movies, TV Series, or specific content types

#### Stage 2: Exposure Modeling 🎯
- **Exposure Probability Prediction**: Machine learning model predicts what content you've likely seen
- **Calibrated Classification**: Accounts for exposure bias in recommendation scoring
- **Training Features**: Genre overlap, popularity, release year, vote counts

#### Stage 3: Preference Modeling ❤️
- **Pairwise Learning**: Learns preferences from rating comparisons when possible
- **Personal Score Prediction**: Estimates how much you'd like each candidate
- **Fallback Strategy**: Uses exposure probabilities when insufficient rating data

#### Stage 4: Final Scoring & Ranking 📊
- **Z-Score Normalization**: Balances personal preferences with popularity
- **Weighted Combination**: `final_score = user_weight * personal_z + global_weight * popularity_z`
- **MMR Diversity**: Ensures diverse recommendations using Maximum Marginal Relevance
- **Content Filtering**: Applies content type restrictions if specified

#### Traditional Approaches (Also Available)

**Popularity-based Recommender:**
- Calculates content similarity between seed titles and candidates
- Weights by user preference patterns and global popularity
- Applies recency bias if specified

**SVD Matrix Factorization:**
- Builds user-item matrix from ratings
- Applies singular value decomposition with optimal component selection
- Handles cold-start problems with global priors

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

## Testing

This project includes a comprehensive test suite with 30+ tests covering all major functionality. The test suite is designed to be fast, reliable, and comprehensive.

### Test Suite Overview

**Current Status:** ✅ 30/30 tests passing  
**Test Coverage:** Comprehensive coverage of all core modules  
**Runtime:** < 3 seconds for full suite

### Test Categories

1. **Functionality Tests** (`test_functionality.py`) - 16 tests
   - Feature engineering (genres, year buckets, content vectors)
   - Data ingestion and normalization
   - PopSim and SVD recommenders
   - Ranking and blending algorithms
   - CLI integration
   - End-to-end workflow validation
   - Error handling and edge cases

2. **Logger Tests** (`test_logger.py`) - 1 test
   - Action logging functionality
   - CSV format validation
   - Idempotency guarantees

3. **Performance Tests** (`test_performance.py`) - 5 tests
   - Data ingestion performance (< 5s)
   - Recommendation generation speed
   - Scalability with large datasets
   - Memory usage validation
   - Multi-instance stress testing

4. **Selenium Integration Tests** (`test_selenium.py`) - 8 tests
   - CSV replay functionality
   - Browser automation setup
   - Security and credential handling
   - Login workflow validation (with mocking)
   - CAPTCHA detection and handling

### Running Tests

#### Quick Start

```bash
# Install test dependencies (first time only)
pip install -e ".[test]"

# Run all tests (recommended)
pytest tests/ -q

# Run comprehensive test validation
python validate_tests.py
```

#### Basic Test Execution

```bash
# Run all tests (quick)
pytest tests/ -q

# Run all tests (verbose)
pytest tests/ -v

# Run specific test file
pytest tests/test_functionality.py -v

# Run specific test
pytest tests/test_functionality.py::TestFeatureEngineering::test_genres_to_vec -v
```

#### Test Coverage

To run tests with coverage analysis, first install pytest-cov:

```bash
pip install pytest-cov
```

Then run coverage analysis:

```bash
# Generate coverage report
pytest tests/ -q --cov=imdb_recommender --cov-report=term-missing

# Generate HTML coverage report
pytest tests/ --cov=imdb_recommender --cov-report=html

# View HTML report
open htmlcov/index.html
```

#### Performance Testing

```bash
# Run only performance tests
pytest tests/test_performance.py -v

# Run tests with timing information
pytest tests/ -v --durations=10
```

### Test Data and Fixtures

The test suite uses carefully crafted fixtures for reliable testing:

- **`fixtures_ratings.csv`**: Sample IMDb ratings data with various edge cases
- **`fixtures_watchlist.csv`**: Sample watchlist data for integration testing
- **Synthetic datasets**: Controlled data for algorithm validation

### Continuous Integration

The test suite is designed to run in CI environments with:

- **No external dependencies**: All tests run offline
- **Deterministic results**: Fixed seeds ensure repeatable outcomes  
- **Fast execution**: Complete suite runs in < 30 seconds
- **Cross-platform compatibility**: Works on macOS, Linux, and Windows

### Test Design Principles

1. **Hermetic**: No network calls, no external services
2. **Fast**: Quick feedback for development cycles  
3. **Comprehensive**: Covers all major code paths and edge cases
4. **Reliable**: Deterministic with no flaky tests
5. **Maintainable**: Clear structure and good documentation

### Adding New Tests

When adding features, include:

- Unit tests for core functionality
- Integration tests for end-to-end workflows
- Edge case and error condition testing
- Performance validation for algorithms
- CLI testing for user interfaces

Example test structure:

```python
def test_new_feature():
    # Arrange
    input_data = create_test_data()
    
    # Act
    result = your_function(input_data)
    
    # Assert
    assert result.is_valid()
    assert result.meets_requirements()
```

### Selenium Testing Notes

Selenium tests include options to skip tests requiring credentials:

```bash
# Skip tests that need IMDb credentials
pytest tests/test_selenium.py -v -k "not credentials"
```

For full Selenium testing, ensure you have:

- Chrome browser installed
- `.env` file with `IMDB_USERNAME` and `IMDB_PASSWORD` (optional)

## Contributing

This is a personal movie recommendation system. Contributions welcome for:

- Additional recommendation algorithms
- Better feature engineering
- UI improvements
- Performance optimizations
- Enhanced test coverage

### Development Workflow

1. Fork the repository
2. Create a feature branch
3. Add comprehensive tests for new functionality
4. Ensure all tests pass: `pytest tests/ -q`
5. Run code formatting: `black .` and `ruff check .`
6. Submit a pull request

## License

MIT License - see LICENSE file for details.
