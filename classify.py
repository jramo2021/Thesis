# import numpy as np
# import os
# import PIL
# import PIL.Image
# import tensorflow as tf
# import pathlib
# import matplotlib.pyplot as plt
# import support_functions as support
from main import *

# Example from https://www.tensorflow.org/tutorials/load_data/images

def train(train_ds,val_ds,class_names):
    '''Train the Model'''
    num_classes = 2

    model = tf.keras.Sequential([
        tf.keras.layers.Rescaling(1./255),
        tf.keras.layers.Conv2D(32, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(32, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(32, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(num_classes)
    ])


    # To view the training and validation accuracy for each training epock, 
    # pass the metrics argument to model.compile
    model.compile(
        optimizer='adam',
        loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'])

    # Fit data to the model
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=3
    )

    return model

    # '''DATA AUGMENTATION'''
    # '''https://www.tensorflow.org/tutorials/images/data_augmentation
    #   Increases the diversity of your training set by applying random 
    #   (but realistic) transformations, such as image rotation'''


    # # # Create a `Counter` object and `Dataset.zip` it together with the training set.
    # # counter = tf.data.experimental.Counter()
    # # train_ds = tf.data.Dataset.zip((train_datasets, (counter, counter)))

    # # Shuffle and augment the Training dataset
    # train_ds = (
    #     train_ds
    #     .shuffle(1000)
    #     .map(support.augment, num_parallel_calls=AUTOTUNE)
    #     .batch(batch_size)
    #     .prefetch(AUTOTUNE)
    # )

    # # Shuffle and augment the Validation dataset
    # val_ds = (
    #     val_ds
    #     .map(support.resize_and_rescale, num_parallel_calls=AUTOTUNE)
    #     .batch(batch_size)
    #     .prefetch(AUTOTUNE)
    # )

    # # Shuffle and augment the Tesing dataset
    # test_ds = (
    #     test_ds
    #     .map(support.resize_and_rescale, num_parallel_calls=AUTOTUNE)
    #     .batch(batch_size)
    #     .prefetch(AUTOTUNE)
    # )

    # support.visualize_sample_data(class_names,train_ds,'augmented_training_set')