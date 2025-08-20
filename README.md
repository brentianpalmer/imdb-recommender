# 🎬 IMDb Personal Recommender

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A comprehensive movie and TV show recommendation system that learns your personal taste from your IMDb ratings and watchlist. Features both **SVD collaborative filtering** and **ElasticNet feature engineering** approaches with rigorous scientific validation.

## 🏆 Performance Results

| Method | RMSE | Std Dev | Best Parameters | Approach |
|--------|------|---------|----------------|----------|
| **🥇 ElasticNet** | **1.387** | **0.094** | α=0.1, l1_ratio=0.1 | Feature Engineering |
| 🥈 SVD | 1.618 | 0.053 | 24 factors, reg=0.05 | Collaborative Filtering |

**Winner: ElasticNet by 14.3%** - Feature engineering with rich metadata beats pure collaborative filtering.

## 🚀 Key Features

### 🎯 Two Powerful Approaches
1. **ElasticNet with Feature Engineering**
   - 106 engineered features (genres, directors, temporal patterns)
   - Superior accuracy: 1.387 RMSE
   - Handles cold start scenarios
   - Interpretable feature importance

2. **SVD Collaborative Filtering** 
   - Pure user-item interaction patterns
   - Optimized: 24 factors, 0.05 regularization, 20 iterations
   - Stable performance: 1.618 RMSE ± 0.053
   - Scalable matrix factorization

### 🛠️ Core Functionality
- **🎬 Personalized Recommendations**: Learns from your actual IMDb ratings
- **📊 Content Filtering**: Movies, TV Series, Documentaries, etc.
- **� Smart Explanations**: Understand why items were recommended
- **💾 Data Export**: CSV exports with predictions and metadata
- **🖥️ CLI Interface**: Simple command-line tools
- **🧪 Scientific Validation**: Proper cross-validation, no data leakage

## 📋 Quick Start

### Installation
```bash
# Clone repository
git clone https://github.com/brentianpalmer/imdb-recommender.git
cd imdb-recommender

# Install package
pip install -e .

# Install ML dependencies
pip install scikit-learn
```

### Data Setup
1. Export your IMDb ratings and watchlist as CSV
2. Place in `data/raw/` directory:
   - `data/raw/ratings.csv` (IMDb ratings export)
   - `data/raw/watchlist.xlsx` (IMDb watchlist export)
3. Configure `config.toml` if needed

### Generate Recommendations
```bash
# Quick recommendations using SVD
imdbrec recommend --seeds tt0111161 --topk 10

# Feature-rich ElasticNet approach
python run_elasticnet_cv.py --ratings_file data/raw/ratings.csv

# Compare both methods
python validate_comparison.py
```

## 🔬 Scientific Validation

### Methodology
- **ElasticNet**: 5-fold stratified cross-validation
- **SVD**: 3-fold cross-validation (corrected for data leakage)
- **No Data Leakage**: Complete train/test separation
- **Statistical Significance**: Results validated across multiple folds

### Performance Analysis
```python
# ElasticNet Results
Best RMSE: 1.387 ± 0.094 (n=5 folds)
Features: 106 engineered features
Configuration: α=0.1, l1_ratio=0.1 (90% Ridge, 10% Lasso)

# SVD Results  
Best RMSE: 1.618 ± 0.053 (n=3 folds)
Configuration: 24 factors, 0.05 regularization, 20 iterations
Matrix: (3 × 1066) with hybrid user-global weighting
```

### Data Integrity Note
> **⚠️ Important**: Earlier claims of 0.54 RMSE were due to data leakage in cross-validation.
> See [`docs/DATA_LEAKAGE_ANALYSIS.md`](docs/DATA_LEAKAGE_ANALYSIS.md) for full analysis of the issue and correction.

## � Feature Engineering (ElasticNet)

The ElasticNet approach uses 106 carefully engineered features:

### Content Features
- **Genres**: Multi-hot encoded (Action, Drama, Comedy, etc.)
- **Directors**: Top-30 most frequent directors as one-hot features
- **Title Types**: Movie, TV Series, Documentary, etc.
- **Decades**: Release decade groupings

### Temporal Features  
- **Rating Patterns**: Days since first rating, rating frequency
- **Release Timing**: Month of release, age at time of rating
- **Behavioral Signals**: Rating day of week, seasonal patterns

### Numerical Features
- **Content Metadata**: Year, runtime, IMDb rating, vote counts
- **Derived Features**: Log-transformed vote counts, rating age
- **Statistical Features**: Z-score normalized within cross-validation folds

## 🎯 When to Use Each Approach

### Choose ElasticNet When:
- ✅ Rich metadata is available  
- ✅ Cold start scenarios (new users/items)
- ✅ Feature interpretability is important
- ✅ Maximum prediction accuracy is needed
- ✅ Traditional ML pipeline integration

### Choose SVD When:
- ✅ Pure collaborative filtering is desired
- ✅ Minimal metadata is available
- ✅ Recommendation systems at scale
- ✅ Focus on user-item interaction patterns
- ✅ Matrix factorization approach preferred

## 🗂️ Project Structure

```
imdb_recommender_pkg/
├── 🎯 Core Implementation
│   ├── imdb_recommender/          # Main package
│   │   ├── recommender_svd.py     # SVD collaborative filtering
│   │   ├── cross_validation.py    # Validation framework
│   │   └── cli.py                 # Command-line interface
│   └── run_elasticnet_cv.py       # ElasticNet feature engineering
│
├── 📊 Analysis & Validation
│   ├── fine_tune_svd_corrected.py # Corrected SVD validation
│   ├── validate_comparison.py     # Method comparison
│   └── test_elasticnet.py         # ElasticNet testing
│
├── 📋 Documentation
│   └── docs/
│       ├── ALL_IN_ONE_GUIDE.md             # Complete usage guide
│       ├── ELASTICNET_VS_SVD_COMPARISON.md # Performance comparison
│       ├── DATA_LEAKAGE_ANALYSIS.md        # Validation correction
│       └── INTEGRATION_SUMMARY.md          # Implementation summary
│
├── 📁 Data
│   ├── data/raw/                  # Raw IMDb exports
│   └── data/                      # Processed datasets
│
└── 📈 Results
    └── results/
        ├── elasticnet_cv_results.csv  # ElasticNet grid search
        └── svd_corrected_results.json # SVD validation results
```

## 🚀 Advanced Usage

### Custom ElasticNet Grid Search
```bash
python run_elasticnet_cv.py \
  --ratings_file data/raw/ratings.csv \
  --alphas 0.01,0.1,1.0,10.0 \
  --l1_ratios 0.1,0.5,0.9 \
  --n_splits 5 \
  --top_dir_k 50 \
  --out_csv custom_results.csv
```

### SVD Hyperparameter Validation
```bash
python fine_tune_svd_corrected.py
```

### Performance Comparison
```bash
python validate_comparison.py
```

## 📈 Results Interpretation

### ElasticNet Feature Importance
- **Top predictors**: IMDb rating, genre preferences, director familiarity
- **Temporal patterns**: Rating behavior over time, seasonal preferences  
- **Content signals**: Runtime preferences, decade biases
- **Regularization**: 90% Ridge (feature stability) + 10% Lasso (selection)

### SVD Factor Analysis
- **Latent dimensions**: 24 factors capture user-item interaction patterns
- **Regularization**: 0.05 prevents overfitting to sparse data
- **Hybrid approach**: 70% user ratings + 30% global popularity
- **Convergence**: 20 iterations optimal for stability vs. performance

## 🔍 Troubleshooting

### Common Issues
1. **Missing data files**: Ensure `data/raw/ratings.csv` exists with IMDb export format
2. **Import errors**: Install dependencies with `pip install -e . && pip install scikit-learn`
3. **Memory issues**: ElasticNet may require 4GB+ RAM for feature engineering
4. **Performance differences**: Results may vary with different train/test splits

### Data Format Requirements
```csv
# ratings.csv (IMDb export format)
Const,Your Rating,Date Rated,Title,Genres,Directors,Year,IMDb Rating,...
tt0111161,9,2024-01-15,The Shawshank Redemption,"Drama",Frank Darabont,1994,9.3,...
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make changes with proper validation
4. Run tests: `python validate_comparison.py`
5. Submit pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📚 Scientific References

- **Matrix Factorization**: Koren, Y., Bell, R., & Volinsky, C. (2009). Matrix factorization techniques for recommender systems.
- **ElasticNet Regularization**: Zou, H., & Hastie, T. (2005). Regularization and variable selection via the elastic net.
- **Cross-Validation**: Arlot, S., & Celisse, A. (2010). A survey of cross-validation procedures for model selection.

---

**🎯 Bottom Line**: Choose ElasticNet for maximum accuracy with rich metadata, or SVD for scalable collaborative filtering. Both approaches are scientifically validated and production-ready.

*Last updated: August 20, 2025 | Dataset: 544 IMDb ratings | Validation: Rigorous cross-validation without data leakage*
