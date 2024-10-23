#!/usr/bin/env python3
# classify.py
import argparse
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras import models
import os


def decode_batch_predictions(pred, max_len, symbols):
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    results = tf.keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0]
    decoded = tf.keras.backend.get_value(results)

    predictions = []
    for seq in decoded:
        label = ''.join([symbols[int(i)] for i in seq if i != -1])
        predictions.append(label)
    return predictions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--width', type=int, required=True, help='Width of captcha images')
    parser.add_argument('--height', type=int, required=True, help='Height of captcha images')
    parser.add_argument('--length', type=int, required=True, help='Maximum captcha length (<= 6)')
    parser.add_argument('--captcha-dir', type=str, required=True,
                        help='Directory containing captcha images to classify')
    parser.add_argument('--input-model', type=str, required=True, help='Path to the trained model file')
    parser.add_argument('--symbols', type=str, required=True, help='File containing the symbols used in captchas')
    parser.add_argument('--output', type=str, required=True, help='File where the classifications should be saved')

    args = parser.parse_args()

    # Validate length argument
    if args.length <= 0 or args.length > 6:
        print("Error: Length must be between 1 and 6.")
        exit(1)

    # Check if captcha directory exists
    if not os.path.exists(args.captcha_dir):
        print(f"Error: Captcha directory '{args.captcha_dir}' does not exist.")
        exit(1)

    # Check if the model file exists
    if not os.path.exists(args.input_model):
        print(f"Error: Model file '{args.input_model}' does not exist.")
        exit(1)

    # Check if the symbols file exists
    if not os.path.exists(args.symbols):
        print(f"Error: Symbols file '{args.symbols}' does not exist.")
        exit(1)

    # Load captcha symbols
    with open(args.symbols) as symbols_file:
        captcha_symbols = symbols_file.readline().strip()

    # Load the model
    model = models.load_model(args.input_model)

    with open(args.output, 'w') as output_file:
        for file_name in os.listdir(args.captcha_dir):
            # Load and preprocess the input image
            raw_data = cv2.imread(os.path.join(args.captcha_dir, file_name))
            if raw_data is None:
                print(f"Warning: Could not read image {file_name}. Skipping.")
                continue

            rgb_data = cv2.cvtColor(raw_data, cv2.COLOR_BGR2RGB)
            processed_data = np.array(rgb_data) / 255.0
            processed_data = processed_data.reshape((1, args.height, args.width, 3))

            # Classify the image
            predictions = model.predict(processed_data)
            decoded_predictions = decode_batch_predictions(predictions, args.length, captcha_symbols)

            # Write the classification result to the output file
            output_file.write(f"{file_name}, {decoded_predictions[0]}\n")
            print(f'Classified {file_name} as {decoded_predictions[0]}')


if __name__ == '__main__':
    main()
    #classify
