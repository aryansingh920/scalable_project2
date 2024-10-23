#!/usr/bin/env python3
# generate.py
import os
import numpy
import random
import string
import cv2
import argparse
from captcha.image import ImageCaptcha
from PIL import ImageFont

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--width', help='Width of captcha image', type=int)
    parser.add_argument('--height', help='Height of captcha image', type=int)
    parser.add_argument('--length', help='Length of captchas in characters', type=int)
    parser.add_argument('--count', help='How many captchas to generate', type=int)
    parser.add_argument('--output-dir', help='Where to store the generated captchas', type=str)
    parser.add_argument('--symbols', help='File with the symbols to use in captchas', type=str)
    parser.add_argument('--font', help='Path to the font file to use', type=str)
    args = parser.parse_args()

    if args.width is None:
        print("Please specify the captcha image width")
        exit(1)

    if args.height is None:
        print("Please specify the captcha image height")
        exit(1)

    if args.length is None:
        print("Please specify the captcha length")
        exit(1)

    if args.count is None:
        print("Please specify the captcha count to generate")
        exit(1)

    if args.output_dir is None:
        print("Please specify the captcha output directory")
        exit(1)

    if args.symbols is None:
        print("Please specify the captcha symbols file")
        exit(1)

    if args.font is None:
        print("Please specify the font file")
        exit(1)

    # Initialize the captcha generator with specified font
    captcha_generator = ImageCaptcha(width=args.width, height=args.height, fonts=[args.font])

    # Read the captcha symbols from the file
    with open(args.symbols, 'r') as symbols_file:
        captcha_symbols = symbols_file.readline().strip()

    print("Generating captchas with symbol set {" + captcha_symbols + "}")

    # Check if the output directory exists, if not create it
    if not os.path.exists(args.output_dir):
        print("Creating output directory " + args.output_dir)
        os.makedirs(args.output_dir)

    generated_count = 0
    while generated_count < args.count:
        # Generate a random captcha string
        random_str = ''.join([random.choice(captcha_symbols) for j in range(random.randint(1, args.length))])
        image_path = os.path.join(args.output_dir, random_str + '.png')

        # Check if the file already exists
        if os.path.exists(image_path):
            print(f"Captcha {random_str}.png already exists, skipping...")
            continue  # Try generating a different captcha

        # Generate the captcha image and save it using OpenCV
        image = numpy.array(captcha_generator.generate_image(random_str))
        cv2.imwrite(image_path, image)
        print(f"Captcha {random_str}.png generated successfully")
        generated_count += 1

if __name__ == '__main__':
    main()
