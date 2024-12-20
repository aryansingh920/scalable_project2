#!/bin/bash
#run.sh

python3 -m venv venv
source ./venv/bin/activate

pip3 install -r requirements.txt

#to download all files from csv
python3 src/access_files/download_files.py

#to compare and get a csv files not downloaded to download the files again
python3 src/access_files/compareCsv.py

# to generate
python3 src/image_processing/generate_process/generate.py --width 192 --height 96 --length 6 --symbols src/image_processing/symbols.txt --count 3200 --output-dir src/data/test --font src/image_processing/font/font1.ttf



#to divide to training data and validation_data
python3 src/image_processing/generate_process/divide.py


# to train
python3 src/image_processing/train/train.py --train-dataset "src/data/Aryan/captchas_test" --validate-dataset "src/data/Vishalini/captchas_test" --batch-size 64 --epochs 12 --output-model-name "src/image_processing/captcha_model.keras" --length 6 --width 192 --height 96 --symbols "src/image_processing/symbols.txt"



# to predict 
python3 src/image_processing/classify/classify.py --width 192 --height 96 --length 6 --captcha-dir src/data/Aryan/captchas_test --input-model src/image_processing/captcha_model.keras.keras --symbols src/image_processing/symbols.txt --output src/image_processing/output/output.txt



#to convert txt to csv
python3 src/image_processing/generate_process/convert_to_csv.py