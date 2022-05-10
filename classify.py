from main import *
import sys
import tensorflow as tf
import time

# Example from https://www.tensorflow.org/tutorials/load_data/images
def define_model():

    # '''Create model (Original)'''
    # model = tf.keras.Sequential([
    #     tf.keras.layers.Rescaling(1./255),
    #     tf.keras.layers.Conv2D(32, 3, activation='relu'),
    #     tf.keras.layers.MaxPooling2D(),
    #     tf.keras.layers.Conv2D(32, 3, activation='relu'),
    #     tf.keras.layers.MaxPooling2D(),
    #     tf.keras.layers.Conv2D(32, 3, activation='relu'),
    #     tf.keras.layers.MaxPooling2D(),
    #     tf.keras.layers.Flatten(),
    #     tf.keras.layers.Dense(128, activation='relu'),
    #     tf.keras.layers.Dense(2)
    # ])

    '''Model 1.1'''
    model = tf.keras.Sequential([
        tf.keras.layers.Rescaling(1./255),
        tf.keras.layers.Conv2D(32, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(2)
    ])

    # '''Model 1.2'''
    # model = tf.keras.Sequential([
    #     tf.keras.layers.Rescaling(1./255),
    #     tf.keras.layers.Conv2D(32, 3, activation='relu'),
    #     tf.keras.layers.MaxPooling2D(),
    #     tf.keras.layers.Flatten(),
    #     tf.keras.layers.Dense(256, activation='relu'),
    #     tf.keras.layers.Dense(2)
    # ])

    # '''Model 1.3'''
    # model = tf.keras.Sequential([
    #     tf.keras.layers.Rescaling(1./255),
    #     tf.keras.layers.Conv2D(64, 3, activation='relu'),
    #     tf.keras.layers.MaxPooling2D(),
    #     tf.keras.layers.Flatten(),
    #     tf.keras.layers.Dense(128, activation='relu'),
    #     tf.keras.layers.Dense(2)
    # ])

    # '''Model 1.4'''
    # model = tf.keras.Sequential([
    #     tf.keras.layers.Rescaling(1./255),
    #     tf.keras.layers.Conv2D(32, 3, activation='relu'),
    #     tf.keras.layers.MaxPooling2D(),
    #     tf.keras.layers.Dropout(0.2),
    #     tf.keras.layers.Flatten(),
    #     tf.keras.layers.Dense(128, activation='relu'),
    #     tf.keras.layers.Dense(2)
    # ])
    
    
    

    # '''ResNet model'''
    # model = tf.keras.applications.resnet50.ResNet50()

    # '''VGG16 model'''
    # model = tf.keras.applications.vgg16.VGG16(
    #     include_top=True,
    #     weights=None,
    #     classes=2)
    

    # To view the training and validation accuracy for each training epoch, 
    # pass the metrics argument to model.compile
    # print("\nCompiling Model")
    
    '''Compile the Model'''
    model.compile(
        optimizer='adam',
        loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
        # loss=tf.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=['accuracy','mae'])

    return model

def train(model,train_ds,val_ds):
    
    # Start Training Timer
    start = time.time()

 
    '''Fit data to model using 3 epochs'''
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=25
    )

    
    # '''Fit data to model using 25 epochs and early stopping'''
    # # https://www.geeksforgeeks.org/choose-optimal-number-of-epochs-to-train-a-neural-network-in-keras/
    # # Early Stopping Call back will stop training if the model 
    # # hasn't decreased the validation loss in the last 5 epochs.
    # # If it stops early, it will restore the weights that had the minimum loss.
    # earlystopping = tf.keras.callbacks.EarlyStopping(
    #     monitor ="val_loss", 
    #     mode ="min", 
    #     patience = 5, 
    #     restore_best_weights = True)

    # if not os.path.isdir(path):
    #     # Create the checkpoint path in the Docker container
    #     os.makedirs(path)

    # # https://keras.io/api/callbacks/model_checkpoint/
    # # Save the best model 
    # model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    #     # filepath = path,
    #     # filepath=path+'model.{epoch:02d}-{val_loss:.2f}.h5',
    #     filepath=path+'checkpoint.h5',
    #     save_weights_only=True,
    #     monitor='val_accuracy',
    #     mode='max',
    #     save_best_only=True)



    # # Start Training Timer
    # start = time.time()

    # # Fit Training Data to Model (Train Data)
    # history = model.fit(
    #     train_ds,
    #     validation_data=val_ds,        
    #     epochs=25,
    #     # callbacks =[earlystopping,model_checkpoint_callback]
    #     # callbacks =[model_checkpoint_callback]
    # )

    # # Reload best weights from the file
    # model.load_weights(path+'checkpoint.h5')


    # print(model.evaluate(val_ds))

    end = time.time()
    
    training_time = end-start
    print("\nTraining took:%0.2f" %training_time,"seconds\n")

    return model, history, training_time
    

    