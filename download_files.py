import csv
import os
import asyncio
import httpx
from time import sleep

# Asynchronous function to download PNG files with retries, backoff, and logging


async def download_png(client, download_link, save_folder, filename, retry_limit=5, backoff_factor=2):
    try:
        retries = 0
        backoff = 1
        while retries < retry_limit:
            response = await client.get(download_link)
            if response.status_code == 200:
                file_path = os.path.join(save_folder, filename)
                with open(file_path, 'wb') as file:
                    file.write(response.content)
                print(f"Downloaded {filename} successfully.")
                return True  # Download successful
            else:
                retries += 1
                print(f"Failed to download {filename} (Attempt {
                      retries}/{retry_limit}) - Status: {response.status_code}")
                await asyncio.sleep(backoff)
                backoff *= backoff_factor  # Exponential backoff for each retry

        print(f"Failed to download {filename} after {retry_limit} attempts.")
        return False  # Download failed
    except Exception as e:
        print(f"Error downloading {filename}: {e}")
        return False  # Download failed

# Asynchronous function to handle all downloads with limited concurrency


async def download_files_from_csv(csv_file_path, save_folder, shortname_variable, max_concurrent_downloads=10):
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)  # Create folder if it doesn't exist

    # Read the CSV file without column names
    with open(csv_file_path, mode='r') as file:
        # Treat each row as a single file name
        csv_reader = [row[0] for row in csv.reader(file)]
        total_files = len(csv_reader)  # Total number of files
        downloaded_count = 0  # Counter for downloaded files
        failed_files = []  # List to track failed downloads

        # Define headers to mimic a browser visit
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }

        # Create an asynchronous client session
        async with httpx.AsyncClient(headers=headers, timeout=None) as client:
            tasks = []
            # Limit concurrent requests
            semaphore = asyncio.Semaphore(max_concurrent_downloads)

            async def limited_download(filename):
                async with semaphore:
                    download_link = f"https://cs7ns1.scss.tcd.ie/?shortname={
                        shortname_variable}&myfilename={filename}"
                    success = await download_png(client, download_link, save_folder, filename)
                    return success

            for filename in csv_reader:
                tasks.append(limited_download(filename))

            # Wait for all tasks to complete with concurrency limits
            results = await asyncio.gather(*tasks)

            # Count successful downloads
            downloaded_count = sum(results)
            failed_files = [csv_reader[i]
                            for i, result in enumerate(results) if not result]

        # Log failed downloads
        if failed_files:
            with open('failed_downloads.txt', 'w') as fail_log:
                fail_log.write("\n".join(failed_files))

        # Print progress
        print(f"\nDownload complete. Successfully downloaded {
              downloaded_count} out of {total_files} files.")
        if failed_files:
            print(f"Failed downloads logged in 'failed_downloads.txt'.")

# Main function to run the async loop



def main():
    # Define paths and variables
    csv_file_path = 'singha12-challenge-filenames.csv'  # CSV file path
    save_folder = 'test_captcha'  # Folder where files will be saved
    shortname_variable = 'singha12'  # Replace with your actual shortname value

    # Download files
    asyncio.run(download_files_from_csv(
        csv_file_path, save_folder, shortname_variable))

# Entry point
if __name__ == "__main__":
    main()




