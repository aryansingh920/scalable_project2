#!/usr/bin/env python3
import os
import random
import argparse
import cv2
import numpy as np
import tensorflow.keras as keras
from tensorflow.keras import layers
import tensorflow as tf


# CTC loss function
def ctc_loss_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    return tf.keras.backend.ctc_batch_cost(labels, y_pred, input_length, label_length)


# CNN + RNN + CTC model architecture
def create_model(max_captcha_length, captcha_num_symbols, input_shape, rnn_units=128):
    input_tensor = layers.Input(shape=input_shape)

    # Convolutional layers
    x = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(input_tensor)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    # Flatten and reshape for RNN layers
    x = layers.Reshape((-1, x.shape[-1]))(x)

    # RNN layers (LSTM)
    x = layers.Bidirectional(layers.LSTM(rnn_units, return_sequences=True))(x)

    # Dense layer with softmax activation for character prediction
    x = layers.Dense(captcha_num_symbols, activation='softmax')(x)

    # Model for training
    model = keras.Model(inputs=input_tensor, outputs=x)

    return model


# Custom data generator with support for variable-length captchas
class CaptchaSequence(keras.utils.Sequence):
    def __init__(self, directory, batch_size, max_captcha_length, symbols, width, height):
        self.directory = directory
        self.batch_size = batch_size
        self.max_captcha_length = max_captcha_length
        self.symbols = symbols
        self.width = width
        self.height = height
        self.files = os.listdir(directory)

    def __len__(self):
        return len(self.files) // self.batch_size

    def __getitem__(self, idx):
        batch_files = self.files[idx * self.batch_size:(idx + 1) * self.batch_size]
        X = np.zeros((self.batch_size, self.height, self.width, 3), dtype=np.float32)
        labels = np.ones([self.batch_size, self.max_captcha_length]) * -1  # Padding with -1 for CTC
        label_lengths = np.zeros([self.batch_size, 1])
        input_lengths = np.ones([self.batch_size, 1]) * (self.width // 4)

        for i, file_name in enumerate(batch_files):
            label = file_name.split('.')[0]
            image = cv2.imread(os.path.join(self.directory, file_name))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0
            X[i] = image

            # Encode labels and handle variable-length captchas
            for j, ch in enumerate(label):
                labels[i, j] = self.symbols.find(ch)
            label_lengths[i] = len(label)

        inputs = [X, labels, input_lengths, label_lengths]
        outputs = np.zeros([self.batch_size])
        return inputs, outputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--width', type=int, required=True)
    parser.add_argument('--height', type=int, required=True)
    parser.add_argument('--length', type=int, required=True, help='Maximum captcha length')
    parser.add_argument('--batch-size', type=int, required=True)
    parser.add_argument('--epochs', type=int, required=True)
    parser.add_argument('--train-dataset', type=str, required=True)
    parser.add_argument('--validate-dataset', type=str)
    parser.add_argument('--symbols', type=str, required=True)
    parser.add_argument('--output-model', type=str, required=True)
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

    if args.batch_size is None:
        print("Please specify the training batch size")
        exit(1)

    if args.epochs is None:
        print("Please specify the number of training epochs to run")
        exit(1)

    if args.train_dataset is None:
        print("Please specify the path to the training data set")
        exit(1)

    if args.validate_dataset is None:
        print("Please specify the path to the validation data set")
        exit(1)

    if args.output_model_name is None:
        print("Please specify a name for the trained model")
        exit(1)

    if args.symbols is None:
        print("Please specify the captcha symbols file")
        exit(1)

    captcha_symbols = None



    with open(args.symbols, 'r') as f:
        captcha_symbols = f.readline().strip()

    # Build the model
    model = create_model(args.length, len(captcha_symbols), (args.height, args.width, 3))

    # CTC loss calculation
    labels = layers.Input(name='the_labels', shape=[args.length], dtype='float32')
    input_lengths = layers.Input(name='input_length', shape=[1], dtype='int64')
    label_lengths = layers.Input(name='label_length', shape=[1], dtype='int64')
    loss_out = layers.Lambda(ctc_loss_lambda_func, output_shape=(1,), name='ctc')(
        [model.output, labels, input_lengths, label_lengths])

    model_ctc = keras.Model(inputs=[model.input, labels, input_lengths, label_lengths], outputs=loss_out)
    model_ctc.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer='adam')

    # Training and validation data generators
    train_gen = CaptchaSequence(args.train - dir, args.batch_size, args.length, captcha_symbols, args.width,
                                args.height)
    if args.validate_dir:
        val_gen = CaptchaSequence(args.validate_dir, args.batch_size, args.length, captcha_symbols, args.width,
                                  args.height)
    else:
        val_gen = None

    # Train the model
    model_ctc.fit(train_gen, validation_data=val_gen, epochs=args.epochs)

    # Save the model
    model.save(args.output_model)


if __name__ == '__main__':
    main()
