#!/usr/bin/env python3
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import os
import numpy as np
import cv2
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.utils import to_categorical

# Hardcoded input arguments
train_dataset = 'C:/kv/GitHub/scalable_project2/src/image_processing/train/training_data'
validate_dataset = 'C:/kv/GitHub/scalable_project2/src/image_processing/train/validation_data'
batch_size = 32
epochs = 5
input_model_name = 'captcha_model.h5'
output_model_name = 'captcha_model'
captcha_min_length = 1  # Minimum length of CAPTCHA
captcha_max_length = 7  # Maximum length of CAPTCHA
captcha_width = 192
captcha_height = 96
captcha_symbols = "123456789aBCdeFghjkMnoPQRsTUVwxYZ+%|#][{}\-$%"  # Set of symbols used in CAPTCHA

# Build a Keras model given some parameters
def create_model(captcha_max_length, captcha_num_symbols, input_shape, model_depth=5, module_size=2):
    input_tensor = keras.Input(input_shape)
    x = input_tensor
    for i in range(model_depth):
        for j in range(module_size):
            x = keras.layers.Conv2D(32 * 2**min(i, 3), kernel_size=3, padding='same', kernel_initializer='he_uniform')(x)
            x = keras.layers.BatchNormalization()(x)
            x = keras.layers.Activation('relu')(x)
        x = keras.layers.MaxPooling2D(2)(x)

    x = keras.layers.Flatten()(x)
    # Multi-output for each character in the CAPTCHA
    outputs = [keras.layers.Dense(captcha_num_symbols, activation='softmax', name=f'char_{i+1}')(x) for i in range(captcha_max_length)]
    model = keras.Model(inputs=input_tensor, outputs=outputs)

    return model

# A Sequence represents a dataset for training in Keras
class ImageSequence(keras.utils.Sequence):
    def __init__(self, directory_name, batch_size, captcha_min_length, captcha_max_length, captcha_symbols, captcha_width, captcha_height):
        self.directory_name = directory_name
        self.batch_size = batch_size
        self.captcha_min_length = captcha_min_length
        self.captcha_max_length = captcha_max_length
        self.captcha_symbols = captcha_symbols
        self.captcha_width = captcha_width
        self.captcha_height = captcha_height
        self.num_symbols = len(captcha_symbols)

        if not os.path.exists(self.directory_name):
            print(f"Directory {self.directory_name} does not exist.")
            os.makedirs(self.directory_name)

        # List all files in the dataset directory
        self.files = [f for f in os.listdir(self.directory_name) if f.endswith('.png')]
        if not self.files:
            raise ValueError(f"No PNG files found in {self.directory_name}.")

    def __len__(self):
        return int(np.ceil(len(self.files) / self.batch_size))
    
    def __getitem__(self, idx):
        batch_files = self.files[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_images = []
        batch_labels = [[] for _ in range(self.captcha_max_length)]  # List of lists to hold labels for each character position

        for file_name in batch_files:
            image_path = os.path.join(self.directory_name, file_name)
            image = cv2.imread(image_path)
            image = cv2.resize(image, (self.captcha_width, self.captcha_height))
            image = image.astype('float32') / 255.0

            # Extract CAPTCHA text
            captcha_text = file_name.split('.')[0]
            cleaned_text = captcha_text.split('_')[0]  # Ignore anything after "_"
            captcha_length = len(cleaned_text)

            # Assign each character in the CAPTCHA to one output of the model
            for i in range(captcha_length):
                char_label = self.captcha_symbols.index(cleaned_text[i])
                batch_labels[i].append(to_categorical(char_label, num_classes=self.num_symbols))

            # If CAPTCHA is shorter than max length, append blank labels
            for i in range(captcha_length, self.captcha_max_length):
                batch_labels[i].append(np.zeros(self.num_symbols))

            batch_images.append(image)

        batch_images = np.array(batch_images)

        # Convert labels to numpy arrays
        batch_labels = [np.array(label) for label in batch_labels]

        if len(batch_images) == 0 or not all([len(label) > 0 for label in batch_labels]):
            raise ValueError(f"Empty batch or empty labels at index {idx}.")

        return batch_images, tuple(batch_labels)

# Main function for building, training, and saving the model
def main():
    strategy = tf.distribute.MirroredStrategy()  # Use multi-GPU if available
    print("Number of devices: {}".format(strategy.num_replicas_in_sync))

    with strategy.scope():
        model = create_model(captcha_max_length, len(captcha_symbols), (captcha_height, captcha_width, 3))
        
        if os.path.exists(input_model_name):
            print(f"Loading weights from {input_model_name}")
            model.load_weights(input_model_name)

        # Assign unique accuracy metrics for each character output
        metrics = {f'char_{i+1}': 'accuracy' for i in range(captcha_max_length)}

        # Compile the model
        model.compile(
            loss='categorical_crossentropy',
            optimizer=keras.optimizers.Adam(1e-3, amsgrad=True),
            metrics=metrics  # Unique accuracy for each output
        )

    model.summary()

    # Data generators for training and validation sets
    training_data = ImageSequence(train_dataset, batch_size, captcha_min_length, captcha_max_length, captcha_symbols, captcha_width, captcha_height)
    validation_data = ImageSequence(validate_dataset, batch_size, captcha_min_length, captcha_max_length, captcha_symbols, captcha_width, captcha_height)

    # Callbacks for training
    callbacks = [
        keras.callbacks.EarlyStopping(patience=3),
        keras.callbacks.ModelCheckpoint(output_model_name + '.keras', save_best_only=True)
    ]

    # Save the model architecture to JSON
    with open(output_model_name + ".json", "w") as json_file:
        json_file.write(model.to_json())

    # Train the model
    model.fit(
        training_data,
        validation_data=validation_data,
        epochs=epochs,
        callbacks=callbacks,
    )

if __name__ == '__main__':
    main()
