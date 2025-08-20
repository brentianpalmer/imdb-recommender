#!/usr/bin/env python3
"""
Quick test of ElasticNet CV on a smaller parameter grid to verify functionality
"""
import subprocess


def main():
    print("🧪 QUICK TEST: ElasticNet CV with smaller grid")
    print("=" * 50)

    # Run with smaller grid for quick validation
    cmd = [
        "/Users/brent/workspace/imdb_recommender_pkg/.venv/bin/python",
        "run_elasticnet_cv.py",
        "--ratings_file",
        "data/raw/ratings.csv",
        "--n_splits",
        "3",
        "--alphas",
        "0.1,1.0",
        "--l1_ratios",
        "0.1,0.9",
        "--top_dir_k",
        "10",
        "--out_csv",
        "elasticnet_test.csv",
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        print("✅ Command executed successfully")
        print("STDOUT:", result.stdout[-200:])  # Last 200 chars
        if result.stderr:
            print("STDERR:", result.stderr[-200:])
        print(f"Return code: {result.returncode}")

        if result.returncode == 0:
            print("🎉 ElasticNet CV test passed!")
        else:
            print("❌ ElasticNet CV test failed")

    except subprocess.TimeoutExpired:
        print("⏰ Test timed out after 60 seconds")
    except Exception as e:
        print(f"❌ Test error: {e}")


if __name__ == "__main__":
    main()
