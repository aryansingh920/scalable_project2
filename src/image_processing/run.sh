#!/bin/bash


python3 -m venv .venv
source ./.venv/bin/activate

pip3 install -r requriements.txt


# to generate
python src/image_processing/generate.py --width 128 --height 64 --length 4 --symbols src/image_processing/symbols.txt --count 32 --output-dir src/image_processing/test --font src/image_processing/font/font1.ttf


# to train
python src/image_processing/train.py --width 128 --height 64 --length 4 --symbols src/image_processing/symbols.txt --batch-size 32 --epochs 5 --output-model src/image_processing/test.h5 --train-dataset src/image_processing/train_data --validate-dataset src/image_processing/validation_data


# to predict 
python src/image_processing/classify.py --model-name src/image_processing/test --captcha-dir src/image_processing/validation_data --output src/output.txt --symbols src/image_processing/symbols.txt
