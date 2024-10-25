#!/bin/bash
#run.sh

python3.10 -m venv venv
source ./venv/bin/activate

pip3.10 install -r requirements.txt


# to generate
python3.10 src/image_processing/generate_process/generate.py --width 192 --height 96 --length 6 --symbols src/image_processing/symbols.txt --count 3200 --output-dir src/data/test --font src/image_processing/font/font1.ttf



#to divide to training data and validation_data
python3.10 src/image_processing/generate_process/divide.py


# to train
python3.10 src/image_processing/train/train.py --train-dataset "src/data/training_data" --validate-dataset "src/data/validation_data" --batch-size 64 --epochs 12 --input-model "src/image_processing/captcha_model.h5" --output-model-name "src/image_processing/captcha_model.keras" --length 6 --width 192 --height 96 --symbols "src/image_processing/symbols.txt"




# to predict 
python3.10 src/image_processing/classify/classify.py --width 192 --height 96 --length 6 --captcha-dir src/data/Aryan/captchas_test --input-model src/image_processing/captcha_model.keras --symbols src/image_processing/symbols.txt --output src/image_processing/output/output.txt
