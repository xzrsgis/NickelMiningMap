# Main script to run preprocessing, training or prediction tasks.
# Convert this to a notebook if you are using jupyter.

# This is where you can change which config to use, by replacing 'config_default' with 'my_amazing_config' etc
# TRAINING
#import os
import tensorflow as tf
tf.config.list_physical_devices('GPU')
print(tf.test.is_built_with_cuda())  # Should return True
print(tf.test.is_gpu_available())  # Should return True if GPU is available