import csv
import requests
import time
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

# Function to download and save content from a URL with retries
def download_content(url, filename, retries=5, delay=2):
    attempt = 0
    while attempt < retries:
        try:
            response = requests.get(url)
            if response.status_code == 200:
                # Ensure directory exists
                os.makedirs(os.path.dirname(filename), exist_ok=True)

                # Write the content to the file
                with open(filename, 'wb') as f:
                    f.write(response.content)
                return True  # Successful download
            else:
                print(f"Failed to download {filename}: HTTP {response.status_code}")
        except Exception as e:
            print(f"Error downloading {filename}: {e}")

        attempt += 1
        print(f"Retrying {filename}... Attempt {attempt} of {retries}")
        time.sleep(delay)  # Wait before retrying

    print(f"Failed to download {filename} after {retries} attempts.")
    return False  # Failed after retries

def process_row(row, base_url, download_directory):
    url_extension = row[0]
    full_url = f"{base_url}{url_extension}"
    # Save in the specified directory
    filename = os.path.join(download_directory, url_extension)
    return download_content(full_url, filename)

# Function to read CSV and run downloads in parallel
def process_csv_parallel(csv_file, base_url, download_directory, max_workers=5):
    with open(csv_file, newline='') as csvfile:
        csvreader = csv.reader(csvfile)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for row in csvreader:
                # Submitting the download task for each row in the CSV
                futures.append(executor.submit(process_row, row, base_url, download_directory))

            # Collect results as they are completed
            for future in as_completed(futures):
                try:
                    result = future.result()  # Get the result of each download task
                except Exception as e:
                    print(f"An error occurred: {e}")

# Main script
if __name__ == "__main__":
    csv_file = r'ramasamv.csv'  # Use raw string for Windows path
    # Replace with your base URL
    base_url = 'https://cs7ns1.scss.tcd.ie/?shortname=ramasamv&myfilename='
    download_directory = 'test_dataset'  # Directory to save downloaded files

    # Process CSV and download files in parallel
    process_csv_parallel(csv_file, base_url, download_directory, max_workers=8)
