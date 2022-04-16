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

    # '''Create model (Mod 2)'''
    # model = tf.keras.Sequential([
    #     tf.keras.layers.Rescaling(1./255),
    #     tf.keras.layers.Conv2D(32, 3, activation='relu'),
    #     tf.keras.layers.MaxPooling2D(),
    #     tf.keras.layers.Flatten(),
    #     tf.keras.layers.Dense(256, activation='relu'),
    #     tf.keras.layers.Dense(num_classes)
    # ])

    # '''Create model (Mod 3)'''
    # model = tf.keras.Sequential([
    #     tf.keras.layers.Rescaling(1./255),
    #     tf.keras.layers.Conv2D(32, 3, activation='relu'),
    #     tf.keras.layers.MaxPooling2D(),
    #     tf.keras.layers.Flatten(),
    #     tf.keras.layers.Dense(128, activation='relu'),
    #     tf.keras.layers.Dense(num_classes)
    # ])

    # '''Create model (Mod 4)'''
    # # Maxed out memory after 5 runs
    # model = tf.keras.Sequential([
    #     tf.keras.layers.Rescaling(1./255),
    #     tf.keras.layers.Conv2D(32, 3, activation='relu'),
    #     tf.keras.layers.MaxPooling2D(),
    #     tf.keras.layers.Flatten(),
    #     tf.keras.layers.Dense(1024, activation='relu'),
    #     tf.keras.layers.Dense(num_classes)
    # ])
    
    # '''Create model (Mod 5)'''
    # # Maxed out memory after 16 runs
    # model = tf.keras.Sequential([
    #     tf.keras.layers.Rescaling(1./255),
    #     tf.keras.layers.Conv2D(32, 3, activation='relu'),
    #     tf.keras.layers.MaxPooling2D(),
    #     tf.keras.layers.Flatten(),
    #     tf.keras.layers.Dense(512, activation='relu'),
    #     tf.keras.layers.Dense(num_classes)
    # ])
    

    # mergedOut = tf.keras.layers.Add()([model0.output,model1.output])
    # model = tf.keras.layers.Model([model0.input,model1.input], mergedOut)

    # # ResNet and KGG16 notes https://keras.io/api/applications/
    # model = tf.keras.Sequential([
    #     tf.keras.applications.ResNet50(
    #         include_top=False,
    #         weights="imagenet",
    #         # input_tensor=None,
    #         input_shape=(256,256,3),
    #         # pooling=None,
    #         classes=num_classes),
    #     tf.keras.layers.Dense(128, activation='relu'),
    #     tf.keras.layers.Dense(num_classes)])

    # # Home made VGG16 https://towardsdatascience.com/step-by-step-vgg16-implementation-in-keras-for-beginners-a833c686ae6c
    # model = tf.keras.Sequential()
    # model.add(tf.keras.layers.Conv2D(input_shape=(224,224,3),filters=64,kernel_size=(3,3),padding="same", activation="relu"))
    # model.add(tf.keras.layers.Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu"))
    # model.add(tf.keras.layers.MaxPool2D(pool_size=(2,2),strides=(2,2)))
    # model.add(tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
    # model.add(tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
    # model.add(tf.keras.layers.MaxPool2D(pool_size=(2,2),strides=(2,2)))
    # model.add(tf.keras.layers.Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
    # model.add(tf.keras.layers.Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
    # model.add(tf.keras.layers.Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
    # model.add(tf.keras.layers.MaxPool2D(pool_size=(2,2),strides=(2,2)))
    # model.add(tf.keras.layers.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    # model.add(tf.keras.layers.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    # model.add(tf.keras.layers.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    # model.add(tf.keras.layers.MaxPool2D(pool_size=(2,2),strides=(2,2)))
    # model.add(tf.keras.layers.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    # model.add(tf.keras.layers.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    # model.add(tf.keras.layers.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    # model.add(tf.keras.layers.MaxPool2D(pool_size=(2,2),strides=(2,2)))

    # model.add(tf.keras.layers.Flatten())
    # model.add(tf.keras.layers.Dense(units=4096,activation="relu"))
    # model.add(tf.keras.layers.Dense(units=4096,activation="relu"))
    # model.add(tf.keras.layers.Dense(units=2, activation="softmax"))
    
    
    
    
    # model = VGG16(weights='imagenet', include_top=False)

    # To view the training and validation accuracy for each training epoch, 
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
    
    training_time = end-start
    print("\nTraining took:%0.2f" %training_time,"seconds\n")

    return model, history, training_time
    

    