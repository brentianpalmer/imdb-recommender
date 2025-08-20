# 🎬 IMDb Personal Recommender - SVD Edition

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A high-performance movie and TV show recommendation system that learns your personal taste from your IMDb ratings and watchlist. Built around an optimized SVD (Singular Value Decomposition) collaborative filtering algorithm.

## 🚀 Key Features

- **🎯 Personalized Recommendations**: Learns from your actual IMDb ratings to predict what you'll enjoy
- **⚡ Optimized SVD Algorithm**: Custom ALS-based SVD with optimal hyperparameters (24 factors, 0.05 regularization, 20 iterations)
- **🎬 Content Filtering**: Get recommendations by content type (Movies, TV Series, Documentaries, etc.)
- **📊 Smart Explanations**: Understand why each recommendation was suggested
- **💾 Data Export**: Export your watchlist and ratings with predictions to CSV
- **🖥️ CLI Interface**: Simple command-line interface for quick recommendations

## 📊 Performance

- **High Accuracy**: Optimized SVD configuration delivers superior prediction quality
- **Fast Inference**: Generate recommendations for 500+ items in seconds
## 🛠️ Installation

### Prerequisites
- Python 3.12+
- Your IMDb ratings exported as CSV
- Your IMDb watchlist exported as CSV

### Install Dependencies
```bash
# Clone the repository
git clone https://github.com/brentianpalmer/imdb-recommender.git
cd imdb-recommender

# Install dependencies
pip install -e .

# Install additional dependencies for validation (optional)
pip install scikit-surprise
```

### Data Setup
1. Export your IMDb ratings and watchlist as CSV files
2. Place them in the `data/` directory as:
   - `data/ratings_normalized.parquet` (your ratings)
   - `data/watchlist_normalized.parquet` (your watchlist)
3. Configure paths in `config.toml`

## 🎯 Quick Start

### Get Top 10 Movie Recommendations
```bash
imdbrec recommend --config config.toml --topk 10 --content-type Movie
```

### Get Top 10 TV Series Recommendations  
```bash
imdbrec recommend --config config.toml --topk 10 --content-type "TV Series"
```

### Get All Recommendations (Mixed Content)
```bash
imdbrec recommend --config config.toml --topk 25
```

### Export Watchlist with Predictions
```bash
python export_watchlist.py
```

## 📁 Project Structure

```
imdb-recommender/
├── imdb_recommender/           # Core recommendation engine
│   ├── cli.py                  # Command-line interface
│   ├── recommender_svd.py      # Optimized SVD algorithm
│   ├── data_io.py              # Data loading and processing
│   ├── config.py               # Configuration management
│   └── features.py             # Content-based features
├── data/                       # Your rating and watchlist data
│   ├── ratings_normalized.parquet
│   └── watchlist_normalized.parquet
├── pretrained_models/          # Optimal hyperparameters
│   └── hyperparameters.json
├── config.toml                 # Configuration file
├── export_watchlist.py         # Export watchlist with predictions
└── validate_custom_svd.py      # Performance validation
```

## ⚙️ Configuration

Edit `config.toml` to customize:

```toml
[paths]
ratings_csv_path = "data/ratings_normalized.parquet"
watchlist_path = "data/watchlist_normalized.parquet"
data_dir = "data"

[runtime]
random_seed = 42
```

## 🧪 Algorithm Details

### SVD Optimization
The recommender uses a custom ALS (Alternating Least Squares) SVD implementation optimized for personal movie recommendations:

- **Factors**: 24 latent factors (optimal for ~500 ratings)
- **Regularization**: 0.05 (low regularization for personal taste learning)
- **Iterations**: 20 (prevents overfitting while ensuring convergence)
- **Multi-User Matrix**: Incorporates both your ratings and IMDb global ratings

### Why SVD Works
- **Collaborative Filtering**: Learns patterns from similar users' preferences
- **Latent Factors**: Discovers hidden taste dimensions (genre preferences, director styles, etc.)
- **Cold Start Handling**: Uses content features and IMDb ratings for new items
- **Personalization**: Tailored specifically to your rating history

## 📊 Data Format

### Expected Ratings Format
```csv
imdb_const,my_rating,title,year,genres,imdb_rating,title_type
tt0111161,10,The Shawshank Redemption,1994,Drama,9.3,Movie
tt0068646,9,The Godfather,1972,Crime Drama,9.2,Movie
```

### Expected Watchlist Format
```csv
imdb_const,title,year,genres,imdb_rating,title_type
tt0468569,The Dark Knight,2008,Action Crime Drama,9.0,Movie
tt0137523,Fight Club,1999,Drama,8.8,Movie
```

## 🔬 Performance Validation

Validate the SVD optimization:
```bash
python validate_custom_svd.py
```

## 📈 Example Output

```
🎬 Top 10 SVD Recommendations:
================================================================================
 1. The Good, the Bad and the Ugly (1966)
    🎯 Score: 0.858  🎬 Adventure, Drama, Western
    💡 predicted high personal rating

 2. One Flew Over the Cuckoo's Nest (1975)
    🎯 Score: 0.847  🎬 Drama
    💡 predicted high personal rating

 3. The Green Mile (1999)
    🎯 Score: 0.836  🎬 Crime, Drama, Fantasy, Mystery
    💡 predicted high personal rating
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **IMDb** for providing the movie database and rating platform
- **scikit-learn** and **NumPy** for machine learning foundations
- **Surprise** library for collaborative filtering benchmarks

## 🔗 Related Projects

- [IMDb Data Processing](docs/DATA_PROCESSING.md) - Data preparation guide
- [Hyperparameter Optimization](REPLICATION_GUIDE.md) - Replication instructions
- [Performance Analysis](validate_custom_svd.py) - Evaluation methodology

---

**Made with ❤️ for movie enthusiasts who want personalized recommendations based on their actual taste.**

- **User Matrix (U)**: Captures user preferences in latent space
- **Item Matrix (V)**: Captures item characteristics in latent space  
- **Prediction**: `rating = U[user] · V[item]`

### Optimized Hyperparameters

Through comprehensive grid search and cross-validation:

- **n_factors**: 8 latent factors (optimal balance of expressiveness vs. overfitting)
- **reg_param**: 0.01 regularization (minimal regularization for best fit)
- **n_iter**: 10 iterations (efficient convergence without overtraining)
- **user_weight**: 0.5 (optimal balance of personal preferences)
- **global_weight**: 0.1 (minimal but beneficial global popularity signal)

### Performance Metrics

Rigorous testing results:

- **RMSE**: 0.828 (excellent rating prediction accuracy)
- **R² Score**: 0.593 (explains 59.3% of rating variance)
- **MAE**: 0.579 (mean absolute error)
- **Training time**: ~663 seconds (acceptable for offline training)
- **Prediction time**: ~0.29 seconds (fast real-time inference)

## Project Structure

```
imdb_recommender_pkg/
├── imdb_recommender/
│   ├── __init__.py
│   ├── cli.py                  # Command line interface
│   ├── config.py              # Configuration management
│   ├── data_io.py             # Data ingestion and processing
│   ├── recommender_svd.py     # Core SVD algorithm
│   ├── recommender_base.py    # Base recommender class
│   ├── ranker.py              # Result ranking and formatting
│   ├── features.py            # Feature engineering utilities
│   ├── hyperparameter_tuning.py # Automated hyperparameter optimization
│   ├── cross_validation.py    # Cross-validation strategies
│   ├── sklearn_integration.py # Scikit-learn compatibility
│   ├── logger.py              # Action logging
│   └── schemas.py             # Data schemas and types
├── tests/                     # Test suite
├── data/                      # Data directory
├── config.toml               # Configuration file
├── pyproject.toml            # Package configuration
└── README.md                 # This file
```

## Testing

Run the test suite to verify functionality:

```bash
# Run all tests
python -m pytest tests/

# Run specific test categories
python -m pytest tests/test_functionality.py  # Core functionality
python -m pytest tests/test_performance.py   # Performance benchmarks
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes and add tests
4. Ensure all tests pass (`python -m pytest`)
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- IMDb for providing the rating and watchlist export functionality
- The collaborative filtering and matrix factorization research community
- Scikit-learn for providing excellent machine learning tools and patterns
