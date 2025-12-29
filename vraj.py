from selenium import webdriver
from selenium.webdriver.common.by import By
import time, pathlib, os
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.firefox.options import Options

TRACKING_FILE = "uploaded_files.txt"

def load_uploaded_files():
    if not os.path.exists(TRACKING_FILE):
        return set()
    with open(TRACKING_FILE, "r") as f:
        return set(line.strip() for line in f if line.strip())

def save_uploaded_file(filename):
    with open(TRACKING_FILE, "a") as f:
        f.write(f"{filename}\n")

options = Options()
options.add_argument("--headless")

driver = webdriver.Firefox(options=options)

driver.get("http://localhost:3000")
wait = WebDriverWait(driver, 20)

folder = "/home/vraj/Downloads/Takeout/Google Photos/Photos from 2025/"
uploaded_files = load_uploaded_files()

# Get all files and their sizes
all_files = []
for file in os.listdir(folder):
    if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.webp', '.mp4', '.mov', '.avi', '.mkv')):
        file_path = os.path.join(folder, file)
        if os.path.isfile(file_path):
            all_files.append((file, os.path.getsize(file_path)))

# Sort files by size descending (highest to lowest)
sorted_files = sorted(all_files, key=lambda x: x[1], reverse=True)

for file, size in sorted_files:
    if file in uploaded_files:
        print(f"Skipping already uploaded file: {file}")
        continue
        
    file_path = str(pathlib.Path(folder) / file)

    print(f"Processing: {file} (Size: {size / (1024*1024):.2f} MB)")

    try:
        # ALWAYS refresh to clear React state and prevent accumulation
        driver.get("http://localhost:3000")
        
        # Click Upload tab
        upload_tab = wait.until(EC.element_to_be_clickable((By.XPATH, "//button[contains(., 'Upload')]")))
        upload_tab.click()
        
        # Wait for the file input
        upload_input = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "input[type='file']")))
        
        print(f"Uploading: {file_path}")
        upload_input.send_keys(file_path)
        
        # Wait for the success icon (increased timeout for LLaVA)
        print(f"Waiting for {file} to finish processing (up to 300s)...")
        wait_long = WebDriverWait(driver, 300)
        wait_long.until(EC.presence_of_element_located((By.CSS_SELECTOR, ".text-green-500")))
        print(f"Successfully uploaded: {file}")
        
        save_uploaded_file(file)
        
    except Exception as e:
        print(f"Error during upload of {file}: {e}")

    # GUARANTEED delay between attempts (successful or not)
    print("Sleeping for 2 seconds before next file...")
    time.sleep(2)

print("Upload process finished. Waiting for final processing...")
time.sleep(5)
driver.quit()