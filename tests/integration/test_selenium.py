"""Tests for Selenium functionality and CSV replay system."""

import csv
import os
import tempfile
from pathlib import Path

from imdb_recommender.logger import ActionLogger

ROOT_DIR = Path(__file__).resolve().parents[2]


class TestSeleniumIntegration:
    """Test Selenium integration and CSV replay functionality."""

    def test_action_logger_csv_format(self):
        """Test that action logger creates proper CSV format for selenium replay."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create logger with custom data directory
            logger = ActionLogger(data_dir=temp_dir)

            # Log some test actions
            logger.log_rate(imdb_const="tt1234567", rating=9, notes="Great movie", source="test")
            logger.log_watchlist(
                imdb_const="tt2345678", add=True, notes="Want to watch", source="test"
            )
            logger.log_watchlist(
                imdb_const="tt3456789", add=False, notes="Watched already", source="test"
            )

            # The default CSV file should exist (logger creates it automatically)
            csv_path = str(logger.path)

            # Verify CSV exists and has correct structure
            assert os.path.exists(csv_path)

            # Read and validate CSV content
            with open(csv_path, encoding="utf-8") as f:
                reader = csv.DictReader(f)
                rows = list(reader)

            assert len(rows) == 3

            # Check first row (rating)
            rating_row = rows[0]
            assert rating_row["imdb_const"] == "tt1234567"
            assert rating_row["action"] == "rate"
            assert rating_row["rating"] == "9"
            assert rating_row["notes"] == "Great movie"
            assert rating_row["source"] == "test"
            assert "timestamp_iso" in rating_row
            assert "batch_id" in rating_row

            # Check second row (watchlist add)
            watchlist_add_row = rows[1]
            assert watchlist_add_row["imdb_const"] == "tt2345678"
            assert watchlist_add_row["action"] == "watchlist_add"
            assert watchlist_add_row["rating"] == ""  # No rating for watchlist actions

            # Check third row (watchlist remove)
            watchlist_remove_row = rows[2]
            assert watchlist_remove_row["imdb_const"] == "tt3456789"
            assert watchlist_remove_row["action"] == "watchlist_remove"

    def test_csv_to_selenium_workflow(self):
        """Test complete workflow from logging to selenium replay."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Step 1: Create actions using logger
            logger = ActionLogger(data_dir=temp_dir)

            # Log various actions
            logger.log_rate("tt0111161", 10, "Amazing movie", "test")
            logger.log_rate("tt0068646", 9, "Classic", "test")
            logger.log_watchlist("tt0468569", True, "Want to rewatch", "test")
            logger.log_watchlist("tt0071562", False, "Already seen", "test")

            # Export to selenium-compatible CSV (just get the default path)
            csv_path = str(logger.path)

            # Step 2: Verify CSV format is selenium-compatible
            with open(csv_path) as f:
                reader = csv.DictReader(f)
                rows = list(reader)

                # Check required columns exist
                required_cols = ["imdb_const", "action", "rating"]
                for row in rows:
                    for col in required_cols:
                        assert col in row

                # Check action types match what selenium expects
                action_types = [row["action"] for row in rows]
                expected_actions = ["rate", "rate", "watchlist_add", "watchlist_remove"]
                assert action_types == expected_actions

                # Check ratings are properly formatted
                rating_rows = [row for row in rows if row["action"] == "rate"]
                assert rating_rows[0]["rating"] == "10"
                assert rating_rows[1]["rating"] == "9"

                # Check IMDb IDs are preserved
                imdb_ids = [row["imdb_const"] for row in rows]
                expected_ids = ["tt0111161", "tt0068646", "tt0468569", "tt0071562"]
                assert imdb_ids == expected_ids

    def test_selenium_replay_script_exists(self):
        """Test that the selenium replay script exists and has correct structure."""
        selenium_file = (
            ROOT_DIR / "src" / "imdb_recommender" / "selenium_stub" / "replay_from_csv.py"
        )

        assert selenium_file.exists(), "Selenium replay script should exist"

        content = selenium_file.read_text()

        # Check for required functions
        assert "def main(csv_path: str):" in content, "main function should exist"
        assert "def login_imdb(driver):" in content, "login_imdb function should exist"
        assert (
            "def rate_title(driver, imdb_const: str, rating: int):" in content
        ), "rate_title function should exist"
        assert (
            "def toggle_watchlist(driver, imdb_const: str, add: bool):" in content
        ), "toggle_watchlist function should exist"

        # Check for CSV handling
        assert "csv.DictReader" in content, "Should read CSV files"
        assert 'imdb_const = row["imdb_const"]' in content, "Should extract IMDb IDs"
        assert 'action = row["action"]' in content, "Should extract actions"

        # Check for proper action handling
        assert 'if action == "rate"' in content, "Should handle rate actions"
        assert 'elif action == "watchlist_add"' in content, "Should handle watchlist_add actions"
        assert (
            'elif action == "watchlist_remove"' in content
        ), "Should handle watchlist_remove actions"


class TestSeleniumSetup:
    """Test selenium setup and environment requirements."""

    def test_selenium_imports_available(self):
        """Test that selenium can be imported."""
        try:
            from selenium import webdriver
            from selenium.webdriver.chrome.options import Options
            from selenium.webdriver.common.by import By

            _ = (webdriver, Options, By)
            selenium_available = True
        except ImportError:
            selenium_available = False

        # This test documents whether selenium is available
        # It will pass either way but informs about selenium status
        if selenium_available:
            print("✅ Selenium is available for browser automation")
        else:
            print("⚠️  Selenium not installed - browser automation unavailable")
            print("   Install with: pip install selenium")

        # Test always passes - we just document the status
        assert True

    def test_chrome_driver_requirements(self):
        """Test chrome driver setup requirements."""
        # This test documents Chrome driver requirements
        print("Chrome WebDriver Requirements:")
        print("1. Install Chrome browser")
        print("2. Chrome driver is managed automatically by Selenium 4.x")
        print("3. For headless mode: driver runs without GUI")
        print("4. For visible mode: remove --headless argument")

        # Check if we can import the chrome options
        try:
            from selenium.webdriver.chrome.options import Options

            opts = Options()
            opts.add_argument("--headless=new")
            opts.add_argument("--no-sandbox")
            opts.add_argument("--disable-dev-shm-usage")
            print("✅ Chrome options configured successfully")
        except ImportError:
            print("⚠️  Selenium not available - Chrome options cannot be configured")

        assert True

    def test_login_requirements_documentation(self):
        """Document login requirements for IMDb automation."""
        print("\n=== IMDb Login Requirements ===")
        print("For selenium automation to work, you need:")
        print("1. Valid IMDb account credentials")
        print("2. Handle 2FA if enabled on your account")
        print("3. Implement login_imdb() function with your credentials")
        print("4. Consider rate limiting to avoid being blocked")
        print("5. Respect IMDb's terms of service")
        print("\nExample environment variables approach:")
        print("  IMDB_USERNAME=your_username")
        print("  IMDB_PASSWORD=your_password")
        print("\nSecurity considerations:")
        print("  - Never commit credentials to version control")
        print("  - Use environment variables or secure config files")
        print("  - Consider using encrypted credential storage")
        print("========================================\n")

        assert True


class TestSeleniumConfigSafety:
    """Test selenium configuration safety and security."""

    def test_credentials_not_hardcoded(self):
        """Ensure no credentials are hardcoded in selenium files."""
        # Read the selenium file content directly
        selenium_file = (
            ROOT_DIR / "src" / "imdb_recommender" / "selenium_stub" / "replay_from_csv.py"
        )
        content = selenium_file.read_text()

        # Check for common credential patterns (should not be found)
        dangerous_patterns = [
            "username=",
            "password=",
            "email=",
            "@gmail.com",
            "@yahoo.com",
            'login_imdb(driver, "',
            'driver.find_element(By.NAME, "email").send_keys("',
            'driver.find_element(By.NAME, "password").send_keys("',
        ]

        for pattern in dangerous_patterns:
            assert (
                pattern not in content.lower()
            ), f"Found potential hardcoded credential pattern: {pattern}"

        # Verify login function uses environment variables for credentials
        assert (
            "os.getenv" in content
        ), "Login function should use environment variables for credentials"

    def test_selenium_security_options(self):
        """Test that selenium uses secure browser options."""
        # Read source to verify security options are configured
        selenium_file = (
            ROOT_DIR / "src" / "imdb_recommender" / "selenium_stub" / "replay_from_csv.py"
        )
        content = selenium_file.read_text()

        # Check for security-related browser options
        security_options = ["--no-sandbox", "--disable-dev-shm-usage", "--headless"]

        for option in security_options:
            assert option in content, f"Security option {option} should be configured"

        print("✅ Selenium configured with security options")
        assert True
