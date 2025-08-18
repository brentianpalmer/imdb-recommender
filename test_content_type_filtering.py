#!/usr/bin/env python3
"""
Test script to thoroughly validate content type filtering functionality.
"""

import os
import subprocess
import tempfile


def run_cli_command(cmd, expect_success=True):
    """Run a CLI command and return the result."""
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

    if expect_success and result.returncode != 0:
        print(f"❌ Command failed: {result.stderr}")
        return None
    elif not expect_success and result.returncode == 0:
        print("❌ Command should have failed but succeeded")
        return None

    print(f"✅ Exit code: {result.returncode}")
    print(f"Output preview: {result.stdout[:200]}...")
    return result


def test_content_type_filtering():
    """Test content type filtering comprehensively."""

    # Base command
    base_cmd = (
        "cd /Users/brent/workspace/imdb_recommender_pkg && "
        "python -m imdb_recommender.cli all-in-one --topk 3 --config config.toml"
    )

    print("\n" + "=" * 80)
    print("🎬 TESTING CONTENT TYPE FILTERING")
    print("=" * 80)

    # Test 1: Movies only
    print("\n1. Testing Movie filtering...")
    result = run_cli_command(f"{base_cmd} --content-type 'Movie'")
    if result and "🎬 Top 3 Movie Recommendations:" in result.stdout:
        print("✅ Movie filtering header correct")
    else:
        print("❌ Movie filtering failed")
        return False

    # Test 2: TV Series only
    print("\n2. Testing TV Series filtering...")
    result = run_cli_command(f"{base_cmd} --content-type 'TV Series'")
    if result and "🎬 Top 3 TV Series Recommendations:" in result.stdout:
        print("✅ TV Series filtering header correct")
    else:
        print("❌ TV Series filtering failed")
        return False

    # Test 3: TV Mini Series only
    print("\n3. Testing TV Mini Series filtering...")
    result = run_cli_command(f"{base_cmd} --content-type 'TV Mini Series'")
    if result and "🎬 Top 3 TV Mini Series Recommendations:" in result.stdout:
        print("✅ TV Mini Series filtering header correct")
    else:
        print("❌ TV Mini Series filtering failed")
        return False

    # Test 4: Watchlist + Content Type combination
    print("\n4. Testing Watchlist + Movie combination...")
    result = run_cli_command(f"{base_cmd} --watchlist-only --content-type 'Movie'")
    if result and "🎯 Top 3 Movie Watchlist Recommendations:" in result.stdout:
        print("✅ Watchlist + Movie combination header correct")
    else:
        print("❌ Watchlist + Movie combination failed")
        return False

    # Test 5: Watchlist + TV Series combination
    print("\n5. Testing Watchlist + TV Series combination...")
    result = run_cli_command(f"{base_cmd} --watchlist-only --content-type 'TV Series'")
    if result and "🎯 Top 3 TV Series Watchlist Recommendations:" in result.stdout:
        print("✅ Watchlist + TV Series combination header correct")
    else:
        print("❌ Watchlist + TV Series combination failed")
        return False

    # Test 6: Invalid content type (should fail)
    print("\n6. Testing invalid content type...")
    result = run_cli_command(f"{base_cmd} --content-type 'Invalid Type'", expect_success=False)
    if result and "Content type 'Invalid Type' not found" in result.stderr:
        print("✅ Invalid content type properly rejected")
    else:
        print("❌ Invalid content type handling failed")
        return False

    # Test 7: Export CSV with content filtering
    print("\n7. Testing CSV export with content filtering...")
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
        csv_path = f.name

    try:
        result = run_cli_command(f"{base_cmd} --content-type 'Movie' --export-csv {csv_path}")
        if result and os.path.exists(csv_path):
            with open(csv_path) as f:
                content = f.read()
                print(f"CSV content preview: {content[:200]}...")
            print("✅ CSV export with content filtering works")
        else:
            print("❌ CSV export with content filtering failed")
            return False
    finally:
        if os.path.exists(csv_path):
            os.unlink(csv_path)

    print("\n" + "=" * 80)
    print("🎉 ALL CONTENT TYPE FILTERING TESTS PASSED!")
    print("=" * 80)
    return True


def test_data_integrity():
    """Test that title_type data is loaded correctly."""

    print("\n" + "=" * 80)
    print("📊 TESTING DATA INTEGRITY")
    print("=" * 80)

    # Test that title_type column is available
    test_script = """
from imdb_recommender.data_io import ingest_sources

# Load data to check title_type availability
res = ingest_sources('data/raw/ratings.csv', 'data/raw/watchlist.xlsx', 'data')
catalog = res.dataset.catalog

if 'title_type' in catalog.columns:
    print("✅ title_type column available")
    types = catalog['title_type'].value_counts()
    print(f"Available types: {types.to_dict()}")
    
    # Verify we have the expected types
    expected_types = ['Movie', 'TV Series', 'TV Mini Series']
    for t in expected_types:
        if t in types.index:
            print(f"✅ {t}: {types[t]} items")
        else:
            print(f"❌ {t}: missing")
            exit(1)
    
    print("✅ Data integrity check passed")
else:
    print("❌ title_type column missing")
    exit(1)
"""

    result = subprocess.run(
        ["python", "-c", test_script],
        cwd="/Users/brent/workspace/imdb_recommender_pkg",
        capture_output=True,
        text=True,
    )

    if result.returncode == 0:
        print(result.stdout)
        return True
    else:
        print(f"❌ Data integrity test failed: {result.stderr}")
        return False


def main():
    """Run all tests."""
    print("🚀 Starting comprehensive content type filtering tests...")

    # Test data integrity first
    if not test_data_integrity():
        print("❌ Data integrity tests failed!")
        return False

    # Test CLI functionality
    if not test_content_type_filtering():
        print("❌ Content type filtering tests failed!")
        return False

    print("\n🎉 ALL TESTS PASSED! Content type filtering is working correctly.")
    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
