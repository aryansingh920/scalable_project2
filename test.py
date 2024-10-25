import cv2
from src.image_processing.train.train_helper import TrainingHelper
import tensorflow as tf

# Load the test image
image_path = 'src/data/Aryan/captchas_test/0a0b46fcdfcaa133d5c1acbd60088f9e0ee36ec4.png'
# Load the input image (use any image containing noise)
image = cv2.imread(image_path)

# Preprocess the image with noise removal, thresholding, and morphological operations
# processed_image = TrainingHelper.preprocess_image(image, blur_method="gaussian", blur_kernel=(15, 15), morph_kernel=(3, 3), morph_iterations=2)
processed_image = TrainingHelper.preprocess_image1(image)

# Call the augment_data function
augmented_image = TrainingHelper.augment_data(image)

# Convert the augmented image back to uint8 format (0-255 range)
augmented_image = tf.clip_by_value(augmented_image, 0, 255)
augmented_image = tf.cast(augmented_image, tf.uint8)

# Display the original and augmented images


# Display the original and preprocessed images
cv2.imshow('Original Image', image)
# Save or display the processed image
cv2.imshow("Processed Image", processed_image)
cv2.imshow('Augmented Image', augmented_image.numpy())  # Convert to numpy array for display

cv2.waitKey(0)
cv2.destroyAllWindows()



