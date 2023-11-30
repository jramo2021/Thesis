import tensorflow as tf
import pandas as pd
from matplotlib import pyplot as plt
import itertools
import copy
import math
import time
from support_functions import *
from data_management import *
from classify import *

total_start = time.time()

#define the model
# base_model = tf.keras.applications.vgg16.VGG16(
#     include_top = False,
#     weights = 'imagenet',
#     input_shape = (224, 224, 3)
# )

base_model = tf.keras.applications.resnet50.ResNet50(
        include_top=False,
        #classes = 2,
        weights = None,
        #weights = 'imagenet',
        input_shape = (224, 224, 3)
)

#base_model.trainable = False

x = base_model.output
x = tf.keras.layers.Flatten()(x)

prediction_layer = tf.keras.Sequential([
        tf.keras.layers.Dense(1024, activation='relu'),
        tf.keras.layers.Dense(2)])(x)

model = tf.keras.Model(inputs=base_model.input, outputs=prediction_layer)

#compile model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-9),
    loss=tf.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=['accuracy','mae']
    )

print(model.summary())

color_mode = 'rgb'
img_size = 224

#load dataset
train_ds = tf.keras.utils.image_dataset_from_directory(
    '/tmp/.keras/TrainTestValData/Train',
    color_mode = color_mode,
    seed=123,
    image_size=(img_size, img_size),
    batch_size=None,
    shuffle=True)
    
test_ds = tf.keras.utils.image_dataset_from_directory(
    '/tmp/.keras/TrainTestValData/Test',
    color_mode = color_mode,
    seed=123,
    image_size=(img_size, img_size),
    batch_size=None,
    shuffle=True)

val_ds = tf.keras.utils.image_dataset_from_directory(
    '/tmp/.keras/TrainTestValData/Val',
    color_mode = color_mode,
    seed=123,
    image_size=(img_size, img_size),
    batch_size=None,
    shuffle=True)

preprocessing = tf.keras.Sequential([tf.keras.layers.Rescaling(1./255),
                                    tf.keras.layers.Normalization()])

train_ds = train_ds.map(lambda x, y: (preprocessing(x, training=True), y),num_parallel_calls=tf.data.AUTOTUNE)
test_ds = test_ds.map(lambda x, y: (preprocessing(x, training=True), y),num_parallel_calls=tf.data.AUTOTUNE)
val_ds = val_ds.map(lambda x, y: (preprocessing(x, training=True), y),num_parallel_calls=tf.data.AUTOTUNE)

# rot_flip_zoom_aug = tf.keras.Sequential([
#     tf.keras.layers.RandomFlip("horizontal",seed=123),
#     tf.keras.layers.RandomRotation(factor=0.35,seed=123),
#     tf.keras.layers.RandomZoom(height_factor = (-0.3, 0), seed=123)
#     ])

# tmp_ds = train_ds.map(
# lambda x, y: (rot_flip_zoom_aug(x, training=True), y),num_parallel_calls=tf.data.AUTOTUNE)

# #visualize_augmentations(train_ds, rot_flip_zoom_aug)

# train_ds = train_ds.concatenate(tmp_ds)

train_ds.shuffle(len(train_ds))
test_ds.shuffle(len(test_ds))
val_ds.shuffle(len(val_ds))

train_ds = train_ds.batch(32)
test_ds = test_ds.batch(32)
val_ds = val_ds.batch(32)

#print('After Adding Augmentation:')
    
print('#Total Batches:',len(train_ds) + len(val_ds) + len(test_ds))
print('#Training Batches:',len(train_ds))
print('#Valdiation Batches:',len(val_ds))
print('#Testing Batches:',len(test_ds),'\n')


AUTOTUNE = tf.data.AUTOTUNE

test_ds_copy = iter(test_ds)

# Train model many times because it is non deterministic
results = []
best_accuracy = 0
num_iterations = 50
for i in range(num_iterations):
    print("\n_______Test",i+1,"_______")
    

    # Create copy of the datasets so that it can be reused
    test_ds, test_ds_copy = itertools.tee(test_ds_copy)

    # Train the Model
    model, history, training_time = train(model, train_ds, val_ds)

    # Start testing timer
    start = time.time()
    
    # Extract True Labels from dataset and Predict Labels from images
    true_labels, predicted_labels = predict_labels(model,test_ds)
    print(true_labels[:31],'\n',predicted_labels[:31])
    
    # Calculates TP, TN, FP, FN and total accuracy
    results.append(calculate_metrics(true_labels, predicted_labels))
    
    # Include training time and test prediction and evaluation times
    end = time.time()
    test_time = end - start
    results[i].extend([training_time, test_time])
    
    # Display an estimate for how much longer the testing will take
    time_estimate(training_time, test_time, (num_iterations-i-1))

    # Keep track of the best performing model. 
    if results[i][0] > best_accuracy:
        best_accuracy = results[i][0]
        best_model = model
        best_history = history
    #re define model
    model = tf.keras.Model(inputs=base_model.input, outputs=prediction_layer)

    #compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-9),
        loss=tf.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=['accuracy','mae']
        )

'''Display Results'''
# Prints out Performance Metrics from the prediction of the test dataset 
display_pred_results(results)

# Plots the distribution of accuracies
plot_results(results)

# Plots the Loss and Accuracy over number of epochs
plot_history(best_history)

# Print Final Time
total_end = time.time()
print('Total Time: %0.2f' % (total_end - total_start),"seconds")