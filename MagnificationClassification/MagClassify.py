import tensorflow as tf
import pandas as pd
from matplotlib import pyplot as plt
import itertools
import copy
import math
import time
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
from AugmentData import rotate_and_flip_images
import cv2
import os

def main():
    total_start = time.time()
    
    vgg16 = tf.keras.applications.vgg16.VGG16(
        include_top = False,
        weights = 'imagenet',
        input_shape = (224, 224, 3)
    )

    ResNet = tf.keras.applications.resnet50.ResNet50(
            include_top=False,
            #classes = 2,
            #weights = None,
            weights = 'imagenet',
            input_shape = (224, 224, 3)
    )

    Inception = tf.keras.applications.inception_v3.InceptionV3(
            include_top=False,
            weights='imagenet',
            #pooling = 'None',
            input_shape=(224, 224, 3)
)

    # for layer in ResNet.layers[:143]:
    #     layer.trainable = False

    #ResNet.trainable = False

    #InceptionV3 Classifier
    # inputs = tf.keras.Input(shape = (224, 224, 3))
    # x = Inception(inputs, training = True)
    # x = tf.keras.layers.Flatten()(x)
    # x = tf.keras.layers.Dense(512, activation = 'relu')(x)
    # outputs = tf.keras.layers.Dense(1, activation = 'sigmoid')(x)
    # model = tf.keras.Model(inputs, outputs)

    #ResNet50 Classifier
    # inputs = tf.keras.Input(shape = (224, 224, 3))
    # x = ResNet(inputs, training = True)
    # x = tf.keras.layers.Flatten()(x)
    # x = tf.keras.layers.Dense(512, activation = 'relu')(x)
    # outputs = tf.keras.layers.Dense(1, activation = 'sigmoid')(x)
    # model = tf.keras.Model(inputs, outputs)

    #Modified model 1.1
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(32,3, activation = 'relu'))
    model.add(tf.keras.layers.MaxPooling2D())
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(256, activation='relu'))
    model.add(tf.keras.layers.Dense(1, activation = 'sigmoid'))

    #Residual Model 1.1
    # inputs = tf.keras.Input(shape = (224, 224, 3))
    # x = tf.keras.layers.Conv2D(32, 3, activation = 'relu')(inputs)
    # x = resblock(x, 3, 32)
    # x = resblock(x, 3, 32)
    # x = tf.keras.layers.MaxPooling2D()(x)
    # x = tf.keras.layers.Flatten()(x)
    # x = tf.keras.layers.Dense(512, activation = 'relu')(x)
    # x = tf.keras.layers.Dense(256, activation = 'relu')(x)
    # outputs = tf.keras.layers.Dense(1, activation = 'sigmoid')(x)
    # model = tf.keras.Model(inputs, outputs)

    #Concatenation Skip Connections
    # inputs = tf.keras.Input(shape = (224, 224, 3))
    # x = tf.keras.layers.Conv2D(32, 3, activation = 'relu')(inputs)
    # x = skipblock(x, 3, 32)
    # x = tf.keras.layers.Flatten()(x)
    # x = tf.keras.layers.Dense(512, activation = 'relu')(x)
    # outputs = tf.keras.layers.Dense(1, activation = 'sigmoid')(x)
    # model = tf.keras.Model(inputs, outputs)

    #Modified model 1.0
    # model = tf.keras.Sequential([
    #     tf.keras.layers.Conv2D(32, 3, activation='relu'),
    #     tf.keras.layers.MaxPooling2D(),
    #     tf.keras.layers.Conv2D(32, 3, activation='relu'),
    #     tf.keras.layers.MaxPooling2D(),
    #     tf.keras.layers.Conv2D(32, 3, activation='relu'),
    #     tf.keras.layers.MaxPooling2D(),
    #     tf.keras.layers.Flatten(),
    #     tf.keras.layers.Dense(512, activation='relu'),
    #     tf.keras.layers.Dense(1, activation='sigmoid')])

    #Modified model 1.0 Batch Norm
    # model = tf.keras.Sequential([
    #     tf.keras.layers.Conv2D(32, 3, activation='relu'),
    #     tf.keras.layers.MaxPooling2D(),
    #     tf.keras.layers.BatchNormalization(),
    #     tf.keras.layers.Conv2D(32, 3, activation='relu'),
    #     tf.keras.layers.MaxPooling2D(),
    #     tf.keras.layers.BatchNormalization(),
    #     tf.keras.layers.Conv2D(32, 3, activation='relu'),
    #     tf.keras.layers.MaxPooling2D(),
    #     tf.keras.layers.Flatten(),
    #     tf.keras.layers.Dense(256, activation='relu'),
    #     tf.keras.layers.Dense(1, activation='sigmoid')])

    color_mode = 'rgb'
    #img_size = 256
    img_size = 224 #needed for vgg16 and resnet50
    mag = 'All'

    #load dataset
    train_ds = tf.keras.utils.image_dataset_from_directory(
        '/tmp/.keras/datasets/' + mag + '/Train',
        color_mode = color_mode,
        seed=123,
        image_size=(img_size, img_size),
        batch_size=None,
        shuffle=True)
        
    test_ds = tf.keras.utils.image_dataset_from_directory(
        '/tmp/.keras/datasets/' + mag + '/Test',
        color_mode = color_mode,
        seed=123,
        image_size=(img_size, img_size),
        batch_size=None,
        shuffle=True)

    val_ds = tf.keras.utils.image_dataset_from_directory(
        '/tmp/.keras/datasets/' + mag + '/Val',
        color_mode = color_mode,
        seed=123,
        image_size=(img_size, img_size),
        batch_size=None,
        shuffle=True)

    # for image, label in train_ds:
    #     image = tf.keras.applications.resnet.preprocess_input(image)

    # for image, label in val_ds:
    #     image = tf.keras.applications.resnet.preprocess_input(image)

    # for image, label in test_ds:
    #     image = tf.keras.applications.resnet.preprocess_input(image)
    
    preprocessing_layer = tf.keras.Sequential(  tf.keras.layers.Rescaling(scale = 1./127.5, offset = -1)
                                                #tf.keras.layers.Rescaling(scale = 1./255)
    )
    
    train_ds = train_ds.map(lambda x, y: (preprocessing_layer(x), y))
    val_ds = val_ds.map(lambda x, y: (preprocessing_layer(x), y))
    test_ds = test_ds.map(lambda x, y: (preprocessing_layer(x), y))

    train_ds.shuffle(len(train_ds))
    test_ds.shuffle(len(test_ds))
    val_ds.shuffle(len(val_ds))

    train_ds = train_ds.batch(32)
    test_ds = test_ds.batch(32)
    val_ds = val_ds.batch(32)
    
    #compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-8),
        #loss=tf.losses.SparseCategoricalCrossentropy(from_logits=False),
        #loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        loss = tf.keras.losses.BinaryCrossentropy(),
        #loss_weights = class_weights,
        metrics=[tf.keras.metrics.BinaryAccuracy(),tf.keras.metrics.TruePositives(), tf.keras.metrics.TrueNegatives(), tf.keras.metrics.FalsePositives(), tf.keras.metrics.FalseNegatives(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
        )

    print('#Total Batches:',len(train_ds) + len(val_ds) + len(test_ds))
    print('#Training Batches:',len(train_ds))
    print('#Valdiation Batches:',len(val_ds))
    print('#Testing Batches:',len(test_ds),'\n')


    AUTOTUNE = tf.data.AUTOTUNE

    test_ds_copy = iter(test_ds)

    start = time.time()

    # Train the Model
    history = model.fit(
            train_ds,
            validation_data=val_ds,        
            epochs=20,
            shuffle = True
            #class_weight=class_weights
            # callbacks=[earlystopping]
            #callbacks =[model_checkpoint_callback]
        )

    end = time.time()

    training_time = end-start
    print("\nTraining took:%0.2f" %training_time,"seconds\n")

    # Start testing timer
    start = time.time()

    results = model.evaluate(test_ds, batch_size=32)

    # include training time and test prediction and evaluation times
    end = time.time()
    test_time = end - start
    
    # Extract accuracy, loss, validation accuracy, and validation loss from the training history
    accuracy = history.history['binary_accuracy']
    loss = history.history['loss']
    val_accuracy = history.history['val_binary_accuracy']
    val_loss = history.history['val_loss']

    plt.figure(figsize=(8, 6))

    # Plot Loss
    plt.subplot(2, 1, 1)
    epochs = range(1, len(loss) + 1)
    plt.plot(epochs, loss, 'bo-', label='Training Loss')
    plt.plot(epochs, val_loss, 'ro-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Plot Accuracy
    plt.subplot(2, 1, 2)
    plt.plot(epochs, accuracy, 'bo-', label='Training Accuracy')
    plt.plot(epochs, val_accuracy, 'ro-', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('accuracy_loss_vs_epoch.png')  # Save plot as an image
    plt.close()

    # Print Final Time
    total_end = time.time()
    
    accuracy = results[1] * 100

    print("Metrics:")
    print("Accuracy: %0.2f" % accuracy)
    print("True Positives: %0.2f" % results[2])
    print("True Negatives: %0.2f" % results[3])
    print("False Positives: %0.2f" % float(results[4]))
    print("False Negatives: %0.2f" % float(results[5]))
    print("Precision: %0.2f" % float(results[6]))
    print("Recall %0.2f" % float(results[7]))
    #print("F1 Score: %0.2f" % 2 * (results[6] * results[7]) / (results[6] + results[7]) )
    print("Training Time: %0.2f" % training_time)
    print('Total Time: %0.2f' % (total_end - total_start),"seconds")     

def resblock(x, kernelsize, filters):
    fx = tf.keras.layers.Conv2D(filters, kernelsize, activation='relu', padding = 'same')(x)
    fx = tf.keras.layers.BatchNormalization()(fx)
    fx = tf.keras.layers.Conv2D(filters, kernelsize, padding = 'same')(fx)
    out = tf.keras.layers.Add()([x,fx])
    out = tf.keras.layers.ReLU()(out)
    out = tf.keras.layers.BatchNormalization()(out)
    return out

def skipblock(x, kernelsize, filters):
    fx = tf.keras.layers.Conv2D(filters, kernelsize, activation='relu', padding = 'same')(x)
    fx = tf.keras.layers.BatchNormalization()(fx)
    fx = tf.keras.layers.Conv2D(filters, kernelsize, padding = 'same')(fx)
    out = tf.keras.layers.Concatenate()([x,fx])
    out = tf.keras.layers.ReLU()(out)
    out = tf.keras.layers.BatchNormalization()(out)
    return out

if __name__ == '__main__': 
    main()