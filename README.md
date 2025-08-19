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
   - **Hyperparameter optimization**: Automated model tuning with cross-validation
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

The main CLI command is `imdbrec` with several subcommands:

#### 1. Data Ingestion

Process your IMDb exports:

```bash
imdbrec ingest --ratings data/raw/ratings.csv --watchlist data/raw/watchlist.xlsx
```

#### 2. Generate Recommendations (⭐ Recommended)

Get personalized movie recommendations from your watchlist:

```bash
# Basic recommendations with top 10 from watchlist
imdbrec recommend --config config.toml --topk 10

# Custom weights for personalization vs popularity
imdbrec recommend --config config.toml --topk 10 --user-weight 0.8 --global-weight 0.2

# Add recency bias for newer content
imdbrec recommend --config config.toml --topk 10 --recency 0.1

# Get help on all options
imdbrec recommend --help
```

**Parameters:**

- `--config`: Path to configuration file (recommended)
- `--topk`: Number of recommendations to return (default: 25)
- `--user-weight`: Weight for personal preferences (default: 0.7)
- `--global-weight`: Weight for popularity (default: 0.3)  
- `--recency`: Bias toward newer releases (default: 0.0)
- `--exclude-rated`: Exclude already-rated items (default: true)

#### 3. Hyperparameter Tuning 🔬

Optimize your recommendation models using scikit-learn's GridSearchCV:

```bash
# Tune all models with default settings
imdbrec hyperparameter-tune --config config.toml --models all

# Tune specific models with custom settings
imdbrec hyperparameter-tune --config config.toml --models pop-sim,svd --cv-folds 5 --test-size 0.2

# Quick tuning for faster results
imdbrec hyperparameter-tune --config config.toml --models all-in-one --cv-folds 3 --test-size 0.3 --quick
```

**What gets tuned with sklearn integration:**

- **AllInOneRecommender**: Exposure model parameters, preference model parameters, candidate pool sizes, diversity settings
- **PopSimRecommender**: Global/user weights, rating thresholds, recency bias  
- **SVDAutoRecommender**: SVD components, regularization, learning rates, iterations

**Output:**

The system creates a `hyperparameter_results/` directory with:

- Model-specific parameter files (`.pkl` and `.json`)
- Comprehensive evaluation metrics using sklearn's cross-validation
- Full tuning summary with model comparison
- Best parameters for each model cached for future use

**Example output:**

```text
🏆 Best Models:
   Lowest RMSE: AllInOneRecommender (2.1420)
   Highest Precision@10: AllInOneRecommender (0.8500)

🎯 Best Parameters (AllInOneRecommender):
   exposure_model_params: {'alpha': 0.01, 'max_iter': 500}
   preference_model_params: {'alpha': 0.001, 'max_iter': 1000}
   candidates: 750
   diversity_lambda: 0.3
```

**Usage Tips:**

- Use 80/20 or 70/30 train/test splits to preserve rating distributions
- Higher `cv-folds` gives more robust results but takes longer
- Start with `--quick` mode, then run full tuning on promising models
- The test set remains untouched during tuning for unbiased evaluation

#### 9. Explain Recommendations

Get detailed explanation for why a specific movie was recommended:

```bash
python -m imdb_recommender.cli explain tt1234567 --ratings data/raw/ratings.csv --watchlist data/raw/watchlist.xlsx
```

## Recent Improvements 🚀

### Scikit-Learn Integration 🤖

- **🔬 GridSearchCV Integration**: Replaced bespoke cross-validation loops with sklearn's industry-standard hyperparameter optimization
- **📊 Cross-Validation Framework**: Standardized CV scoring with sklearn's `cross_validate` function for robust model evaluation  
- **🎯 Custom Recommender Splitter**: Purpose-built train/test splitting that respects temporal ordering and user distributions
- **⚡ Sklearn Estimator Wrappers**: All recommenders now implement sklearn's `BaseEstimator` and `RegressorMixin` interfaces
- **💾 Hyperparameter Caching**: Automated caching system stores tuning results with JSON/pickle persistence
- **🔧 Streamlined CLI**: Clean `imdbrec` command with comprehensive functionality and help system

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
# Get top 10 movies from your watchlist with balanced weights
imdbrec recommend --config config.toml --topk 10 --user-weight 0.8 --global-weight 0.2

# Get recommendations with slight preference for newer content
imdbrec recommend --config config.toml --topk 15 --recency 0.1

# Tune hyperparameters for better recommendations using sklearn
imdbrec hyperparameter-tune --config config.toml --models all --cv-folds 5

# Get help on any command
imdbrec recommend --help
imdbrec hyperparameter-tune --help
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
├── cli.py                      # Command-line interface (imdbrec command)
├── config.py                   # Configuration management
├── data_io.py                  # Data ingestion and processing
├── features.py                 # Feature engineering (genres, years, similarity)
├── logger.py                   # Action logging for Selenium replay
├── ranker.py                   # Recommendation ranking and blending
├── recommender_base.py         # Base recommender class
├── recommender_pop.py          # Popularity-based recommender
├── recommender_svd.py          # SVD matrix factorization recommender
├── recommender_all_in_one.py   # Advanced ML-based recommender
├── hyperparameter_tuning.py    # Hyperparameter optimization with sklearn
├── cross_validation.py         # Cross-validation utilities
├── sklearn_integration.py      # Scikit-learn compatibility layer
├── schemas.py                  # Data models and schemas
└── selenium_stub/              # Selenium automation utilities
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

This project includes a comprehensive test suite with 46+ tests covering all major functionality. The test suite is designed to be fast, reliable, and comprehensive.

### Test Suite Overview

**Current Status:** ✅ 46/46 tests passing  
**Test Coverage:** Comprehensive coverage of all core modules including sklearn integration  
**Runtime:** < 5 seconds for full suite

### Test Categories

1. **Functionality Tests** (`test_functionality.py`) - 16 tests
   - Feature engineering (genres, year buckets, content vectors)
   - Data ingestion and normalization
   - PopSim and SVD recommenders
   - Ranking and blending algorithms
   - CLI integration
   - End-to-end workflow validation
   - Error handling and edge cases

2. **All-in-One Tests** (`test_all_in_one.py`) - 15 tests
   - Advanced ML-based recommendation system
   - Exposure bias modeling
   - Pairwise preference learning
   - Candidate building and filtering
   - Z-score normalization
   - Temporal split evaluation

3. **Cross-Validation Tests** (`test_cross_validation.py`) - 7 tests
   - Stratified train/test splits
   - Temporal ordering preservation
   - User distribution validation
   - sklearn integration compatibility

4. **Hyperparameter Tuning Tests** (`test_hyperparameter_tuning.py`) - 4 tests
   - GridSearchCV integration
   - Parameter caching system
   - Multi-algorithm optimization
   - Performance validation

5. **Performance Tests** (`test_performance.py`) - 5 tests
   - Data ingestion performance (< 5s)
   - Recommendation generation speed
   - Scalability with large datasets
   - Memory usage validation
   - Multi-instance stress testing

6. **Selenium Integration Tests** (`test_selenium.py`) - 8 tests
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
