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

- **Realistic RMSE**: 1.62 ± 0.05 (honest evaluation without data leakage)
- **Optimized Configuration**: 24 factors, 0.05 regularization, 20 iterations  
- **Fast Inference**: Generate recommendations for 500+ items in seconds
- **Validated Methodology**: Proper cross-validation with complete train/test separation

> **⚠️ Important Note**: Earlier claims of 0.54 RMSE were due to data leakage in cross-validation. See [`DATA_LEAKAGE_ANALYSIS.md`](DATA_LEAKAGE_ANALYSIS.md) for details.
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

### Performance Validation
Rigorous cross-validation results (corrected methodology):

- **RMSE**: 1.6179 ± 0.0533 (realistic single-user collaborative filtering performance)
- **R² Score**: -0.0575 (challenging single-user scenario)
- **Validation Method**: 3-fold cross-validation with proper train/test separation
- **Data Leakage**: Eliminated (see [`DATA_LEAKAGE_ANALYSIS.md`](DATA_LEAKAGE_ANALYSIS.md))

### Why SVD Works
- **Collaborative Filtering**: Learns patterns from your rating history
- **Latent Factors**: Discovers hidden taste dimensions (genre preferences, director styles, etc.)
- **Cold Start Handling**: Uses content features and IMDb ratings for new items
- **Personalization**: Tailored specifically to your rating patterns

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

- [Data Leakage Analysis](DATA_LEAKAGE_ANALYSIS.md) - Critical validation methodology correction
- [Fine-tuning Results](fine_tune_svd_corrected.py) - Corrected hyperparameter optimization 
- [Performance Validation](validate_custom_svd.py) - Evaluation methodology

---

**Made with ❤️ for movie enthusiasts who want personalized recommendations based on their actual taste.**
