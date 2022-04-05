from main import *
import sys
import tensorflow as tf
import time

# Example from https://www.tensorflow.org/tutorials/load_data/images

def train(train_ds,val_ds):
    
    start = time.time()
    '''Train the Model'''
    num_classes = 2


    '''Create model (Original)'''
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

    # '''Create model (Mod 1)'''
    # model = tf.keras.Sequential([
    #     tf.keras.layers.Rescaling(1./255),
    #     tf.keras.layers.Conv2D(32, 3, activation='relu'),
    #     tf.keras.layers.MaxPooling2D(),
    #     tf.keras.layers.Flatten(),
    #     tf.keras.layers.Dense(128, activation='relu'),
    #     tf.keras.layers.Dense(num_classes)
    # ])

    # # ResNet and KGG16 notes https://keras.io/api/applications/
    # model = tf.keras.Sequential([
    #     tf.keras.applications.ResNet50(
    #         include_top=False,
    #         weights="imagenet",
    #         # input_tensor=None,
    #         # input_shape=None,
    #         # pooling=None,
    #         classes=num_classes),
    #     tf.keras.layers.Dense(num_classes)])


    # model = VGG16(weights='imagenet', include_top=False)

    # To view the training and validation accuracy for each training epock, 
    # pass the metrics argument to model.compile
    # print("\nCompiling Model")
    
    model.compile(
        optimizer='adam',
        loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy','mae'])

    # # Doesn't Work
    # model.compile(
    #     optimizer='adam',
    #     loss = 'Loss',
    #     # loss=tf.keras.losses.Loss,
    #     # loss = tf.keras.losses.Loss(
    #     #     reduction=losses_utils.ReductionV2.AUTO, name=None),
    #     metrics=['accuracy','mae'])

    # # Terrible Implementation 
    # model.compile(
    #     optimizer='adam',
    #     loss=tf.keras.losses.BinaryCrossentropy(),
    #     metrics=['accuracy','mae'])

    

    '''Fit data to model using 3 epochs'''
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=3
    )

    '''Fit data to model using 25 epochs and early stopping'''
    # # https://www.geeksforgeeks.org/choose-optimal-number-of-epochs-to-train-a-neural-network-in-keras/
    # # Early Stopping Call back will stop training if the model 
    # # hasn't decreased the validation loss in the last 5 epochs.
    # # If it stops early, it will restore the weights that had the minimum loss.
    # earlystopping = tf.keras.callbacks.EarlyStopping(monitor ="val_loss", 
    #                                     mode ="min", patience = 5, 
    #                                     restore_best_weights = True)


    # history = model.fit(
    #     train_ds,
    #     validation_data=val_ds,        
    #     epochs=25,
    #     callbacks =[earlystopping]
    # )

    end = time.time()
    print("\nTraining took:%0.2f" %(end - start),"seconds\n")

    return model, history
    

    