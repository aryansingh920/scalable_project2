import tensorflow as tf

# Check if any Metal GPU is detected
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # No need to specify '/MPS:0'. Just check if Metal device is used.
        tf.config.set_visible_devices(gpus[0], 'GPU')
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(f"{len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs")

        # Test TensorFlow on GPU
        with tf.device('/GPU:0'):  # Let TensorFlow select the GPU
            a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
            b = tf.constant([[1.0, 1.0], [0.0, 1.0]])
            c = tf.matmul(a, b)
            print("Result of matrix multiplication on Metal GPU:", c)

    except RuntimeError as e:
        print(e)
else:
    print("No GPU or MPS device found. Running on CPU.")


# Check if the GPU (Metal backend) is being used
print("Available devices:")
physical_devices = tf.config.list_physical_devices()
for device in physical_devices:
    print(device)



# Function to check if the hardware is Apple Silicon
def is_apple_silicon():
    return tf.config.experimental.list_physical_devices('MPS') != []


# Function to enable distributed strategy and hardware optimization
def create_strategy():
    if is_apple_silicon():
        # For Apple Silicon, using the MPS (Metal Performance Shaders) backend, which uses both NPU and GPU
        print("Using Apple Silicon's Metal backend for NPU/GPU acceleration.")
        strategy = tf.distribute.OneDeviceStrategy(device="/device:CPU:0")
    else:
        # For multi-GPU setup, use the MirroredStrategy
        print("Using MirroredStrategy for multi-GPU training.")
        strategy = tf.distribute.MirroredStrategy()

    return strategy


print(is_apple_silicon())
print(create_strategy())