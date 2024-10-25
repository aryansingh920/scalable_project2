# training_helper.py

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import tensorflow.keras.backend as K

class TrainingHelper:

    @staticmethod
    def preprocess_image(image, blur_method="gaussian", blur_kernel=(5, 5),
                         threshold_method=cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
                         morph_kernel=(2, 2), morph_iterations=1):
        """
        Preprocess the image to remove noise, apply thresholding and morphological operations.

        Parameters:
        - image (numpy array): Input image in BGR format.
        - blur_method (str): Method to use for blurring ('gaussian', 'median', or 'bilateral').
        - blur_kernel (tuple): Kernel size for Gaussian or Median blur. Default is (5, 5).
        - threshold_method (int): Threshold method (e.g., OTSU). Default is cv2.THRESH_BINARY_INV + OTSU.
        - morph_kernel (tuple): Kernel size for morphological operations like erosion/dilation. Default is (2, 2).
        - morph_iterations (int): Number of iterations for morphological operations. Default is 1.

        Returns:
        - preprocessed_image (numpy array): The processed binary image after noise removal, thresholding, and morphological operations.
        """

        # Check if image is a numpy array
        if not isinstance(image, np.ndarray):
            raise TypeError(f"Expected image to be a numpy array, but got {type(image)} instead.")

        # Debugging: Print image shape to ensure it's loaded properly
        print(f"Image shape: {image.shape}")

        # Convert image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply the chosen blurring technique for noise removal
        if blur_method == "gaussian":
            blurred = cv2.GaussianBlur(gray, blur_kernel, 0)
        elif blur_method == "median":
            blurred = cv2.medianBlur(gray, blur_kernel[0])  # Median blur only accepts a single kernel size
        elif blur_method == "bilateral":
            blurred = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)
        else:
            raise ValueError("Invalid blur method. Choose 'gaussian', 'median', or 'bilateral'.")

        # Apply OTSU thresholding
        _, thresh = cv2.threshold(blurred, 0, 255, threshold_method)

        # Morphological operations (erosion and dilation) to clean noise
        morph_kernel = np.ones(morph_kernel, np.uint8)
        morphed = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, morph_kernel, iterations=morph_iterations)

        return morphed



    @staticmethod
    def preprocess_image1(image, blur_kernel=(3, 3), threshold_method=cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
                         morph_kernel=(2, 2), morph_iterations=1):
        """
        Preprocess the image to remove noise, apply thresholding and morphological operations.

        Parameters:
        - image (numpy array): Input image in BGR format.
        - blur_kernel (tuple): Kernel size for Gaussian blur. Default is (3, 3).
        - threshold_method (int): Threshold method (e.g., OTSU). Default is cv2.THRESH_BINARY_INV + OTSU.
        - morph_kernel (tuple): Kernel size for morphological operations like erosion/dilation. Default is (2, 2).
        - morph_iterations (int): Number of iterations for morphological operations. Default is 1.

        Returns:
        - preprocessed_image (numpy array): The processed binary image after thresholding and morphological operations.
        """

        # Convert image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, blur_kernel, 0)

        # Apply OTSU thresholding
        _, thresh = cv2.threshold(blurred, 0, 255, threshold_method)

        # Morphological operations (erosion and dilation)
        morph_kernel = np.ones(morph_kernel, np.uint8)
        morphed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, morph_kernel, iterations=morph_iterations)

        return morphed



    @staticmethod
    def create_rnn_model(input_shape, rnn_units=128, rnn_type='LSTM', num_classes=10, cnn_filters=32,
                         cnn_kernel_size=(3, 3)):
        """
        Create a CNN + LSTM/GRU model for CAPTCHA recognition.

        Parameters:
        - input_shape (tuple): Shape of the input image (height, width, channels).
        - rnn_units (int): Number of units in the LSTM/GRU layer. Default is 128.
        - rnn_type (str): Type of RNN layer to use ('LSTM' or 'GRU'). Default is 'LSTM'.
        - num_classes (int): Number of classes for character recognition.
        - cnn_filters (int): Number of filters in the convolutional layers. Default is 32.
        - cnn_kernel_size (tuple): Kernel size for convolutional layers. Default is (3, 3).

        Returns:
        - model (tf.keras.Model): Compiled model with CNN + RNN.
        """

        inputs = layers.Input(shape=input_shape)

        # CNN layers for feature extraction
        x = layers.Conv2D(cnn_filters, cnn_kernel_size, activation='relu', padding='same')(inputs)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Conv2D(cnn_filters * 2, cnn_kernel_size, activation='relu', padding='same')(x)
        x = layers.MaxPooling2D((2, 2))(x)

        # Reshape for RNN input
        x = layers.Reshape((-1, x.shape[-1]))(x)

        # RNN layer (LSTM or GRU)
        if rnn_type == 'LSTM':
            x = layers.LSTM(rnn_units, return_sequences=True)(x)
        elif rnn_type == 'GRU':
            x = layers.GRU(rnn_units, return_sequences=True)(x)

        # Fully connected layer
        x = layers.TimeDistributed(layers.Dense(num_classes, activation='softmax'))(x)

        # Compile the model
        model = models.Model(inputs=inputs, outputs=x)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        return model



    @staticmethod
    def augment_data(image, rotation_range=10, zoom_range=0.1, noise_factor=0.05):
        """
        Apply data augmentation techniques like rotation, zoom, and noise to the input image.

        Parameters:
        - image (numpy array): Input image to augment.
        - rotation_range (int): Degree range for random rotations. Default is 10.
        - zoom_range (float): Range for random zoom. Default is 0.1 (10% zoom in/out).
        - noise_factor (float): Factor for adding random noise. Default is 0.05.

        Returns:
        - augmented_image (numpy array): The augmented image.
        """

        # Convert to a TensorFlow tensor
        image = tf.convert_to_tensor(image, dtype=tf.float32)

        # Apply random rotation
        image = tf.image.rot90(image, tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32))

        # Apply random zoom
        image = tf.image.resize(image, [int(image.shape[0] * (1 + tf.random.uniform([], -zoom_range, zoom_range))),
                                        int(image.shape[1] * (1 + tf.random.uniform([], -zoom_range, zoom_range)))])

        # Crop back to original size
        image = tf.image.resize_with_crop_or_pad(image, image.shape[0], image.shape[1])

        # Add random noise
        noise = tf.random.normal(shape=tf.shape(image), mean=0.0, stddev=noise_factor, dtype=tf.float32)
        augmented_image = tf.add(image, noise)

        # Clip the pixel values to be between 0 and 255
        augmented_image = tf.clip_by_value(augmented_image, 0.0, 255.0)

        return augmented_image



    @staticmethod
    def ctc_loss(y_true, y_pred):
        """CTC Loss Function"""
        # Ensure the model output has the correct shape
        input_length = K.shape(y_pred)[1]  # [batch_size, time_steps, num_classes]
        label_length = K.shape(y_true)[1]  # [batch_size, num_classes]

        # CTC loss
        return K.ctc_batch_cost(y_true, y_pred, input_length, label_length)

    @staticmethod
    def ctc_accuracy(y_true, y_pred):
        """Custom metric for CTC accuracy."""
        # Decode predicted labels and compare with true labels
        decoded_preds = K.ctc_decode(y_pred, input_length)[0][0]  # Get the decoded output
        # Implement your logic here to calculate accuracy based on decoded_preds and y_true
        return accuracy  # Replace with actual accuracy calculation


