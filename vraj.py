from selenium import webdriver
from selenium.webdriver.common.by import By
import time, pathlib, os
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

driver = webdriver.Firefox()

driver.get("http://localhost:3000/upload")

# Wait for the file input to be present
wait = WebDriverWait(driver, 20)
upload_input = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "input[type='file']")))


folder = "/home/vraj/Downloads/Takeout/Google Photos/Photos from 2025/"

for file in sorted(os.listdir(folder)):
    if not file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.webp', '.mp4', '.mov', '.avi', '.mkv')):
        continue
        
    file_path = str(pathlib.Path(folder) / file)
    print(f"Uploading: {file_path}")

    try:
        upload_input.send_keys(file_path)
        # Wait a bit for the UI to react and start the upload
        time.sleep(2)
    except Exception as e:
        print(f"Error uploading {file}: {e}")

print("Upload process finished. Waiting for final processing...")
time.sleep(10)
driver.quit()