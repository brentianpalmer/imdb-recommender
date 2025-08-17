
from __future__ import annotations
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import csv, time

def login_imdb(driver):
    # TODO: Implement login (username/password/2FA) safely on your machine.
    pass

def rate_title(driver, imdb_const: str, rating: int):
    driver.get(f"https://www.imdb.com/title/{imdb_const}/")
    wait = WebDriverWait(driver, 20)
    try:
        rate_btn = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, "[data-testid='hero-rating-bar__aggregate-rating__score']")))
        rate_btn.click(); time.sleep(1)
        star = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, f"[data-testid='rating-star-{rating}']")))
        star.click(); time.sleep(1)
    except Exception as e:
        print(f"[WARN] Rating failed for {imdb_const}: {e}")

def toggle_watchlist(driver, imdb_const: str, add: bool):
    driver.get(f"https://www.imdb.com/title/{imdb_const}/")
    wait = WebDriverWait(driver, 20)
    try:
        wl_btn = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, "[data-testid='tm-box-wl-button']")))
        wl_btn.click(); time.sleep(1)
    except Exception as e:
        print(f"[WARN] Watchlist toggle failed for {imdb_const}: {e}")

def main(csv_path: str):
    opts = Options(); opts.add_argument("--headless=new"); opts.add_argument("--no-sandbox"); opts.add_argument("--disable-dev-shm-usage")
    driver = webdriver.Chrome(options=opts)
    try:
        login_imdb(driver)
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                imdb_const = row["imdb_const"]; action = row["action"]; rating = row.get("rating")
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
        print("Usage: python replay_from_csv.py data/imdb_actions_log.csv"); sys.exit(1)
    main(sys.argv[1])
