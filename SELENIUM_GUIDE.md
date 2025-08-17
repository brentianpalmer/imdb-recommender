# Selenium Integration Guide

## Overview

The IMDb Recommender package includes Selenium integration for automated browser interactions with IMDb. This allows you to:

1. **Log Actions**: Use the CLI to log movie ratings and watchlist changes
2. **Export CSV**: Generate a CSV file with all your actions  
3. **Replay Actions**: Use Selenium to automatically perform these actions on IMDb

## Prerequisites

### 1. Install Selenium
```bash
pip install selenium
```

### 2. Install Chrome Browser
- Download and install Google Chrome from [chrome.google.com](https://chrome.google.com)
- Chrome WebDriver is automatically managed by Selenium 4.x

### 3. Set Up IMDb Credentials
```bash
# Set environment variables (recommended approach)
export IMDB_USERNAME="your_username"
export IMDB_PASSWORD="your_password"
```

**⚠️ Security Warning**: Never commit credentials to version control!

## Implementation Steps

### Step 1: Implement Login Function
Edit `imdb_recommender/selenium_stub/replay_from_csv.py` and replace the `login_imdb` function:

```python
import os
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

def login_imdb(driver):
    """Login to IMDb using environment variables."""
    username = os.getenv('IMDB_USERNAME')
    password = os.getenv('IMDB_PASSWORD')
    
    if not username or not password:
        raise ValueError("IMDB_USERNAME and IMDB_PASSWORD environment variables required")
    
    # Navigate to IMDb sign in page
    driver.get("https://www.imdb.com/ap/signin")
    
    # Wait for and fill username/email
    wait = WebDriverWait(driver, 10)
    email_field = wait.until(EC.presence_of_element_located((By.NAME, "email")))
    email_field.send_keys(username)
    
    # Fill password
    password_field = driver.find_element(By.NAME, "password")
    password_field.send_keys(password)
    
    # Click sign in button
    sign_in_btn = driver.find_element(By.ID, "signInSubmit")
    sign_in_btn.click()
    
    # Handle 2FA if enabled on your account
    try:
        wait.until(EC.presence_of_element_located((By.NAME, "otpCode")))
        otp_code = input("Enter 2FA code from your authenticator app: ")
        otp_field = driver.find_element(By.NAME, "otpCode")
        otp_field.send_keys(otp_code)
        driver.find_element(By.ID, "auth-signin-button").click()
    except:
        pass  # No 2FA required
    
    # Wait a moment for login to complete
    time.sleep(2)
```

### Step 2: Test the Integration
1. **Log some actions** using the CLI:
   ```bash
   python -m imdb_recommender.cli rate tt0111161 10 --notes "Amazing movie"
   python -m imdb_recommender.cli watchlist add tt0068646
   ```

2. **Export to CSV**:
   ```bash
   python -m imdb_recommender.cli export-log --out actions.csv
   ```

3. **Run Selenium replay**:
   ```bash
   python imdb_recommender/selenium_stub/replay_from_csv.py actions.csv
   ```

### Step 3: Verify Actions on IMDb
- Check your IMDb profile to verify ratings were applied
- Check your watchlist for added/removed items
- Review IMDb activity log for confirmation

## Browser Configuration

### Headless Mode (Default)
The script runs in headless mode (no GUI) by default:
```python
opts.add_argument("--headless=new")
```

### Visible Mode (for Debugging)
To see the browser GUI, comment out the headless option in `replay_from_csv.py`:
```python
# opts.add_argument("--headless=new")  # Comment this line
```

### Security Options
The following security options are pre-configured:
- `--no-sandbox`: Disable sandbox for containerized environments
- `--disable-dev-shm-usage`: Disable /dev/shm usage to prevent crashes

## Best Practices

### 🔒 Security
- Use environment variables for credentials
- Never commit credentials to version control
- Consider using encrypted credential storage
- Enable 2FA on your IMDb account for security

### ⚡ Performance  
- Test with a small batch first (5-10 actions)
- Add delays between actions to avoid rate limiting
- Handle network timeouts gracefully
- Log successful and failed operations

### 🛡️ Safety
- Respect IMDb's terms of service
- Don't perform too many actions too quickly
- Monitor for CAPTCHA or anti-bot measures
- Have a backup of your data

## Troubleshooting

### Common Issues

**"No module named 'selenium'"**
```bash
pip install selenium
```

**"Chrome driver not found"**
- Ensure Chrome browser is installed
- Selenium 4.x manages ChromeDriver automatically

**"Login failed"**
- Check environment variables are set correctly
- Verify your IMDb credentials work manually
- Handle 2FA prompts if enabled

**"Element not found"**
- IMDb may have changed their page structure
- Update CSS selectors in the script
- Add longer wait times for slow networks

**"Too many requests"**
- Add delays between actions: `time.sleep(2)`
- Reduce batch size
- Check if IMDb temporarily blocked your IP

### Testing Without Selenium
You can test the action logging system without Selenium installed:
```bash
python -m pytest tests/test_selenium.py::TestSeleniumIntegration::test_action_logger_csv_format -v
```

## Example Workflow

```bash
# 1. Set up credentials
export IMDB_USERNAME="your_email@example.com"  
export IMDB_PASSWORD="your_secure_password"

# 2. Use the recommender and log actions
python -m imdb_recommender.cli ingest --ratings ratings.csv --watchlist watchlist.csv
python -m imdb_recommender.cli recommend --seeds tt0111161 --topk 10
python -m imdb_recommender.cli rate tt0068646 9 --notes "Classic film"

# 3. Export actions to CSV
python -m imdb_recommender.cli export-log --out my_actions.csv

# 4. Replay on IMDb (after implementing login function)
python imdb_recommender/selenium_stub/replay_from_csv.py my_actions.csv
```

## Support

If you encounter issues:
1. Check the test suite: `python -m pytest tests/test_selenium.py -v`
2. Run the demo: `python demo_selenium.py`
3. Verify your setup matches this guide
4. Test login manually on IMDb first

---

**⚠️ Important**: This tool automates interactions with IMDb. Use responsibly and in accordance with IMDb's terms of service.
