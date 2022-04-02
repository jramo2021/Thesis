from main import *
import sys
import tensorflow as tf
import time

# Example from https://www.tensorflow.org/tutorials/load_data/images

def train(train_ds,val_ds):
    
    start = time.time()
    '''Train the Model'''
    num_classes = 2


    '''Create model'''
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
    # print("\nCompiling Model")
    model.compile(
        optimizer='adam',
        
        loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

        # optimizer='sgd',
        # loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        # # metrics=[tf.keras.metrics.AUC(from_logits=True)])
        
        # metrics=['accuracy'])
        

    # Fit data to the model
    # print("\nFit Model on Training Data")
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=3
        # epochs=4
    )

    end = time.time()
    print("\nTraining took:%0.2f" %(end - start),"seconds\n")

    return model

def rgb_to_grayscale(x):
    #x has shape (batch, width, height, channels)
    return (0.21 * x[:,:,:,:1]) + (0.72 * x[:,:,:,1:2]) + (0.07 * x[:,:,:,-1:])

    

    