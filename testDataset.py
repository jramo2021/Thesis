from main import * 
import os, pathlib
import tensorflow as tf
import time
import matplotlib.pyplot as plt
import numpy as np
import random
import shutil

color_mode = 'rgb'
img_size = 256

tf.debugging.set_log_device_placement(True)
gpus = tf.config.list_physical_devices('GPU')
print(gpus)

a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

# with tf.device('/GPU:0'):
#     train_ds = tf.keras.utils.image_dataset_from_directory(
#             '/tmp/.keras/TrainTestValData/Train',
#             # Try grayscale
#             # color_mode="grayscale",
#             color_mode = color_mode,
#             seed=123,
#             image_size=(img_size, img_size),
#             # image_size=(700, 460),
#             batch_size=32,
#             shuffle=True)

# test_ds = tf.keras.utils.image_dataset_from_directory(
#             '/tmp/.keras/TrainTestValData/Test',
#             # Try grayscale
#             # color_mode="grayscale",
#             color_mode = color_mode,
#             seed=123,
#             image_size=(img_size, img_size),
#             # image_size=(700, 460),
#             batch_size=32,
#             shuffle=True)

# val_ds = tf.keras.utils.image_dataset_from_directory(
#             '/tmp/.keras/TrainTestValData/Val',
#             # Try grayscale
#             # color_mode="grayscale",
#             color_mode = color_mode,
#             seed=123,
#             image_size=(img_size, img_size),
#             # image_size=(700, 460),
#             batch_size=32,
#             shuffle=True)


print('DONE')