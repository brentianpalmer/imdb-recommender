# 🎬 IMDb Recommender: Complete Guide

## Overview

This guide covers both **SVD collaborative filtering** and **ElasticNet feature engineering** approaches for IMDb movie rating prediction and recommendation.

## 🏆 Performance Summary

| Method | RMSE | Approach | Best Use Case |
|--------|------|----------|--------------|
| **ElasticNet** | **1.387** | Feature Engineering | Rich metadata, cold start |
| **SVD** | 1.618 | Collaborative Filtering | User-item interactions, scalability |

## 🚀 Quick Start Commands

### ElasticNet Cross-Validation
```bash
python run_elasticnet_cv.py --ratings_file data/raw/ratings.csv --n_splits 5 --out_csv results/elasticnet_results.csv
```

### SVD Recommendations
```bash
imdbrec recommend --seeds tt0111161 --topk 10 --user-weight 0.7
```

### Compare Both Methods
```bash
python validate_comparison.py
```

## 📊 Key Files

- **`run_elasticnet_cv.py`** - ElasticNet with 106 engineered features
- **`fine_tune_svd_corrected.py`** - SVD validation (data leakage corrected)
- **`validate_comparison.py`** - Side-by-side method comparison
- **`imdb_recommender/cli.py`** - Command-line interface

## 📁 Results Location

All results are stored in `results/`:
- `elasticnet_cv_results.csv` - ElasticNet hyperparameter grid
- `svd_corrected_results.json` - SVD validation results

## 🔍 Documentation

- **[Performance Comparison](ELASTICNET_VS_SVD_COMPARISON.md)** - Detailed analysis
- **[Data Leakage Analysis](DATA_LEAKAGE_ANALYSIS.md)** - Validation correction
- **[Integration Summary](INTEGRATION_SUMMARY.md)** - Implementation details

## 🎯 Bottom Line

**Choose ElasticNet for maximum accuracy** (14.3% better RMSE) when rich metadata is available.  
**Choose SVD for scalable collaborative filtering** when focusing on user-item interaction patterns.

Both methods are scientifically validated with proper cross-validation and no data leakage.