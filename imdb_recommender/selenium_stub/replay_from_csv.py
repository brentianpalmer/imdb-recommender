from __future__ import annotations

import csv
import os
import time

from dotenv import load_dotenv
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

# Load environment variables from .env file
load_dotenv()


def login_imdb(driver):
    """Login to IMDb using credentials from environment variables."""
    username = os.getenv("IMDB_USERNAME")
    password = os.getenv("IMDB_PASSWORD")

    if not username or not password:
        raise ValueError("IMDB_USERNAME and IMDB_PASSWORD environment variables required")

    print(f"Logging in to IMDb as {username}...")

    # Navigate to IMDb main page first (more human-like)
    print("Navigating to IMDb homepage...")
    driver.get("https://www.imdb.com/")
    time.sleep(3)  # Wait like a human would

    # Then go to sign in page
    print("Going to sign-in page...")
    driver.get(
        "https://www.imdb.com/ap/signin?openid.pape.max_auth_age=0&openid.return_to=https%3A%2F%2Fwww.imdb.com%2Fregistration%2Fap-signin-handler%2Fimdb_us&openid.identity=http%3A%2F%2Fspecs.openid.net%2Fauth%2F2.0%2Fidentifier_select&openid.assoc_handle=imdb_us&openid.mode=checkid_setup&siteState=eyJvcGVuaWQuYXNzb2NfaGFuZGxlIjoiaW1kYl91cyIsInJlZGlyZWN0VG8iOiJodHRwczovL3d3dy5pbWRiLmNvbS8ifQ&openid.claimed_id=http%3A%2F%2Fspecs.openid.net%2Fauth%2F2.0%2Fidentifier_select&openid.ns=http%3A%2F%2Fspecs.openid.net%2Fauth%2F2.0"
    )
    wait = WebDriverWait(driver, 15)

    try:
        # Wait for page to load and look for email field
        print("Looking for email field...")
        time.sleep(3)  # Give page time to load

        # Check for CAPTCHA/puzzle first
        try:
            driver.find_element(
                By.CSS_SELECTOR, "[data-cy='captcha'], .cvf-widget, #captchacharacters"
            )
            print("🧩 CAPTCHA/Puzzle detected!")
            print("Please solve the puzzle manually in the browser window.")
            print("Press Enter here after solving the puzzle to continue...")
            input()
        except Exception:
            print("No CAPTCHA detected, proceeding with login...")

        # Try multiple selectors for the email field
        email_field = None
        selectors_to_try = [
            (By.NAME, "email"),
            (By.ID, "ap_email"),
            (By.CSS_SELECTOR, "input[type='email']"),
            (By.CSS_SELECTOR, "input[name='email']"),
        ]

        for selector_type, selector_value in selectors_to_try:
            try:
                email_field = wait.until(
                    EC.presence_of_element_located((selector_type, selector_value))
                )
                print(f"Found email field using selector: {selector_type}='{selector_value}'")
                break
            except Exception:
                continue

        if not email_field:
            print(
                "Could not find email field. Page may have changed or requires manual intervention."
            )
            print("Current page title:", driver.title)
            print("Current URL:", driver.current_url)
            print("Please check the browser window for any challenges or CAPTCHAs.")
            input("Press Enter after resolving any issues to continue...")
            return False

        # Type email slowly like a human
        email_field.clear()
        time.sleep(0.5)
        for char in username:
            email_field.send_keys(char)
            time.sleep(0.1)  # Small delay between keystrokes
        print("Email entered")
        time.sleep(1)

        # Try multiple selectors for password field
        password_field = None
        password_selectors = [
            (By.NAME, "password"),
            (By.ID, "ap_password"),
            (By.CSS_SELECTOR, "input[type='password']"),
            (By.CSS_SELECTOR, "input[name='password']"),
        ]

        for selector_type, selector_value in password_selectors:
            try:
                password_field = driver.find_element(selector_type, selector_value)
                print(f"Found password field using selector: {selector_type}='{selector_value}'")
                break
            except Exception:
                continue

        if not password_field:
            print("Could not find password field. May require manual intervention.")
            input("Press Enter after manually entering password to continue...")
            return False

        # Type password slowly like a human
        password_field.clear()
        time.sleep(0.3)
        for char in password:
            password_field.send_keys(char)
            time.sleep(0.08)  # Small delay between keystrokes
        print("Password entered")
        time.sleep(1.5)

        # Try multiple selectors for sign in button
        sign_in_btn = None
        button_selectors = [
            (By.ID, "signInSubmit"),
            (By.CSS_SELECTOR, "input[type='submit']"),
            (By.CSS_SELECTOR, "button[type='submit']"),
            (By.CSS_SELECTOR, "#signInSubmit"),
        ]

        for selector_type, selector_value in button_selectors:
            try:
                sign_in_btn = driver.find_element(selector_type, selector_value)
                print(f"Found sign in button using selector: {selector_type}='{selector_value}'")
                break
            except Exception:
                continue

        if not sign_in_btn:
            raise Exception("Could not find sign in button with any selector")

        print("Clicking sign in button...")
        sign_in_btn.click()

        # Wait a moment for login to process
        time.sleep(3)

        # Check current URL to see if we hit a puzzle/verification page
        current_url = driver.current_url
        print(f"After login attempt, current URL: {current_url}")

        # Handle Amazon CVF (Customer Verification Form) - the puzzle page
        if "/ap/cvf/request" in current_url:
            print("🧩 AMAZON VERIFICATION PUZZLE DETECTED!")
            print("📝 This is Amazon's customer verification challenge")
            print("👁️ A browser window should be visible showing the puzzle")
            print("✋ Please solve the puzzle manually in the browser window")
            print("⌛ After solving, press Enter here to continue...")
            input()

            # After user solves puzzle, wait a moment and check URL again
            time.sleep(2)
            current_url = driver.current_url
            print(f"After puzzle solving, current URL: {current_url}")

        # Handle 2FA if needed (separate from puzzle)
        try:
            otp_field = driver.find_element(By.NAME, "otpCode")
            print("📱 2FA required - please check your authenticator app")
            otp_code = input("Enter 2FA code: ")
            otp_field.send_keys(otp_code)
            driver.find_element(By.ID, "auth-signin-button").click()
            time.sleep(2)
            print("✅ 2FA code submitted")
        except Exception:
            print("ℹ️ No 2FA required")

        # Final verification - check if we're successfully logged in
        try:
            # Wait a bit more for any redirects
            time.sleep(3)
            current_url = driver.current_url

            # Check if we're on IMDb homepage or any IMDb page (not signin/verification)
            if "imdb.com" in current_url and "/ap/" not in current_url:
                print("✅ Successfully logged in to IMDb!")

                # Double-check by looking for account menu
                try:
                    wait.until(
                        EC.presence_of_element_located(
                            (
                                By.CSS_SELECTOR,
                                "[data-testid='imdb-header-account-menu'], "
                                ".navbar__user, #imdbHeader-account-menu",
                            )
                        )
                    )
                    print("✅ User account menu found - login confirmed!")
                except Exception:
                    print("ℹ️ Account menu not found, but URL suggests login success")

                return True
            else:
                print("⚠️ Still on verification/login page")
                print(f"Current URL: {current_url}")
                print("❌ Login may have failed or requires additional manual steps")
                return False

        except Exception as e:
            print(f"⚠️ Error during login verification: {e}")
            print("🤔 Proceeding anyway - manual verification recommended")
            return False

    except Exception as e:
        print(f"❌ Login failed: {e}")
        return False


def rate_title(driver, imdb_const: str, rating: int):
    driver.get(f"https://www.imdb.com/title/{imdb_const}/")
    wait = WebDriverWait(driver, 20)
    try:
        rate_btn = wait.until(
            EC.element_to_be_clickable(
                (By.CSS_SELECTOR, "[data-testid='hero-rating-bar__aggregate-rating__score']")
            )
        )
        rate_btn.click()
        time.sleep(1)
        star = wait.until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, f"[data-testid='rating-star-{rating}']"))
        )
        star.click()
        time.sleep(1)
    except Exception as e:
        print(f"[WARN] Rating failed for {imdb_const}: {e}")


def toggle_watchlist(driver, imdb_const: str, add: bool):
    driver.get(f"https://www.imdb.com/title/{imdb_const}/")
    wait = WebDriverWait(driver, 20)
    try:
        wl_btn = wait.until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, "[data-testid='tm-box-wl-button']"))
        )
        wl_btn.click()
        time.sleep(1)
    except Exception as e:
        print(f"[WARN] Watchlist toggle failed for {imdb_const}: {e}")


def main(csv_path: str):
    opts = Options()

    # Make browser appear more human-like
    opts.add_argument("--disable-blink-features=AutomationControlled")
    opts.add_argument("--disable-extensions")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument("--disable-gpu")
    opts.add_argument("--remote-debugging-port=9222")

    # Set a realistic user agent
    opts.add_argument(
        "--user-agent="
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )

    # Comment out headless mode to see what's happening
    # opts.add_argument("--headless=new")

    # Additional stealth options
    opts.add_experimental_option("excludeSwitches", ["enable-automation"])
    opts.add_experimental_option("useAutomationExtension", False)

    driver = webdriver.Chrome(options=opts)

    # Execute script to remove webdriver property
    driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")

    # Set additional properties to appear more human
    driver.execute_script(
        """
        Object.defineProperty(navigator, 'languages', {
            get: () => ['en-US', 'en']
        });
        Object.defineProperty(navigator, 'plugins', {
            get: () => [1, 2, 3, 4, 5]
        });
    """
    )
    try:
        login_imdb(driver)
        with open(csv_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                imdb_const = row["imdb_const"]
                action = row["action"]
                rating = row.get("rating")
                if action == "rate" and rating:
                    rate_title(driver, imdb_const, int(float(rating)))
                elif action == "watchlist_add":
                    toggle_watchlist(driver, imdb_const, add=True)
                elif action == "watchlist_remove":
                    toggle_watchlist(driver, imdb_const, add=False)
    finally:
        driver.quit()


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python replay_from_csv.py data/imdb_actions_log.csv")
        sys.exit(1)
    main(sys.argv[1])
