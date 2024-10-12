import csv
import os

# Function to read filenames from the original CSV


def read_filenames_from_csv(csv_file_path):
    with open(csv_file_path, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        filenames = [row[0].strip() for row in csvreader]
    return filenames

# Function to get the list of downloaded files


def get_downloaded_filenames(download_directory):
    return [f for f in os.listdir(download_directory) if os.path.isfile(os.path.join(download_directory, f))]

# Function to compare the two lists and find missing files


def find_missing_files(original_list, downloaded_list):
    return list(set(original_list) - set(downloaded_list))

# Function to save missing files to a CSV file


def save_missing_files_to_csv(missing_filenames, output_csv_file):
    with open(output_csv_file, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        # Write the header
        # csvwriter.writerow(['Missing Filename'])
        # Write the missing filenames
        for filename in missing_filenames:
            csvwriter.writerow([filename])


# Main function
if __name__ == "__main__":
    csv_file_path = r'singha12-challenge-filenames.csv'  # Path to your CSV
    download_directory = 'captchas_test'  # Directory where files are saved
    # Output CSV file to save missing filenames
    output_csv_file = 'missing_files1.csv'

    # Step 1: Get the list of all filenames from the original CSV
    original_filenames = read_filenames_from_csv(csv_file_path)

    # Step 2: Get the list of already downloaded filenames
    downloaded_filenames = get_downloaded_filenames(download_directory)

    # Step 3: Find the missing filenames
    missing_filenames = find_missing_files(
        original_filenames, downloaded_filenames)

    # Step 4: Save the missing filenames to a CSV file
    save_missing_files_to_csv(missing_filenames, output_csv_file)

    # Output the missing files count and save them
    print(f"Missing files ({len(missing_filenames)
                            }) have been saved to {output_csv_file}.")
