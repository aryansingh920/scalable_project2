import csv

# Replace with your actual file paths
input_file = 'src/image_processing/output/output.txt'   # Original .txt file
output_file = 'src/image_processing/output/output.csv' # New .csv file

# Replace with your username
username = "singha12"

# Read and sort the file data
with open(input_file, 'r') as file:
    lines = file.readlines()

# Split each line into filename and captcha, and sort by filename
data = [line.strip().split(',') for line in lines]
data.sort(key=lambda x: x[0])

# Write the sorted data to a new CSV file with a username header
with open(output_file, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow([username])  # First row with username
    # writer.writerow(["Filename", "Captcha"])  # Header row
    writer.writerows(data)  # All sorted data rows

print(f"Data has been successfully written to {output_file}")
