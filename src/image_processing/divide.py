import os
import shutil
import random

# Paths
source_dir = 'test'  # The directory you want to split
test_dir = 'train_data'  # Destination for 90% of the files
validation_dir = 'validation_data'  # Destination for 10% of the files

# Create directories if they don't exist
os.makedirs(test_dir, exist_ok=True)
os.makedirs(validation_dir, exist_ok=True)

# Get all the files in the source directory
files = os.listdir(source_dir)

# Shuffle the files for randomness
random.shuffle(files)

# Calculate the split index
split_index = int(len(files) * 0.9)

# Split the files into test and validation sets
test_files = files[:split_index]
validation_files = files[split_index:]

# Move files to the respective directories
for file in test_files:
    shutil.move(os.path.join(source_dir, file), os.path.join(test_dir, file))

for file in validation_files:
    shutil.move(os.path.join(source_dir, file), os.path.join(validation_dir, file))

print(f'Moved {len(test_files)} files to {test_dir} and {len(validation_files)} files to {validation_dir}')
