#!/bin/bash
#run.sh

python3 -m venv .venv
source ./.venv/bin/activate

pip3 install -r requriements.txt


# to generate
python src/image_processing/generate_process/generate.py --width 192 --height 96 --length 6 --symbols src/image_processing/symbols.txt --count 3200 --output-dir src/data/test --font src/image_processing/font/font1.ttf

#to divide to training data and validation_data
python src/image_processing/generate_process/divide.py


# to train
python src/image_processing/train/train.py --width 192 --height 96 --length 6 --symbols src/image_processing/symbols.txt --batch-size 32 --epochs 1 --output-model src/image_processing/test/test.h5.keras --train-dataset src/data/train/training_data --validate-dataset src/data/train/validation_data


# to predict 
python src/image_processing/classify/classify.py --width 192 --height 96 --length 6 --captcha-dir src/data/Aryan --input-model src/image_processing/test/test.h5.keras --symbols src/image_processing/symbols.txt --output src/image_processing/output/output.txt
