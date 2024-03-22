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

def Vgg16Resnet50():
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
            #weights = None,
            weights = 'imagenet',
            input_shape = (224, 224, 3)
    )

    #base_model.trainable = False

    x = base_model.output
    x = tf.keras.layers.Flatten()(x)

    prediction_layer = tf.keras.Sequential([
            tf.keras.layers.Dense(4096, activation='relu'),
            tf.keras.layers.Dense(4096, activation='relu'),
            tf.keras.layers.Dense(2)])(x)

    model = tf.keras.Model(inputs=base_model.input, outputs=prediction_layer)

    #compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-12),
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
    #for i in range(num_iterations):
    #print("\n_______Test",i+1,"_______")


    # Create copy of the datasets so that it can be reused
    #test_ds, test_ds_copy = itertools.tee(test_ds_copy)

    start = time.time()

    # Train the Model
    history = model.fit(
            train_ds,
            validation_data=val_ds,        
            epochs=1,
            # callbacks=[earlystopping]
            #callbacks =[model_checkpoint_callback]
        )

        # Reload best weights from the file
        # model.load_weights(path+'checkpoint.h5')


    print(model.evaluate(val_ds))

    end = time.time()

    training_time = end-start
    print("\nTraining took:%0.2f" %training_time,"seconds\n")

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

    results.extend([training_time, test_time])

    '''Display Results'''
    # Prints out Performance Metrics from the prediction of the test dataset 
    Display_pred_results(results)

    # Plots the distribution of accuracies
    plot_results(results)

    # Plots the Loss and Accuracy over number of epochs
    plot_history(history)

    # Print Final Time
    total_end = time.time()
    print('Total Time: %0.2f' % (total_end - total_start),"seconds")

def Display_pred_results(results):
    '''Using Accuracy and MAE for Metrics'''
    df = pd.DataFrame(results, columns = ["accuracy","TP","TN","FP","FN","training_time","test_time"])
    
    # Calculate Mean for Performance Metrics
    mean_accuracy = df["accuracy"].mean()*100
    mean_TP = df["TP"].mean()*100
    mean_TN = df["TN"].mean()*100
    mean_FP = df["FP"].mean()*100
    mean_FN = df["FN"].mean()*100
    #mean_train_time = df["training_time"].mean()
    #mean_test_time = df["test_time"].mean()
    mean_precision = ((mean_TP)/(mean_TP + mean_FP)) * 100
    mean_recall = ((mean_TP)/(mean_TP + mean_FN)) * 100
    mean_f1_score = 2 * mean_precision * mean_recall / (mean_precision + mean_recall)
    mean_mcc = ((mean_TP * mean_TN) - (mean_FP * mean_FN))/ ((mean_TP+mean_FN)*(mean_TP+mean_FP)*(mean_FP+mean_TN)*(mean_TN+mean_FN))**(0.5)

    # Calculate Standard Deviation for Performance Metrics
    std_accuracy = df["accuracy"].std()*100
    std_TP = df["TP"].std()*100
    std_TN = df["TN"].std()*100
    std_FP = df["FP"].std()*100
    std_FN = df["FN"].std()*100
    #std_train_time = df["training_time"].std()
    #std_test_time = df["test_time"].std()

    # Calculate Median and Range of Accuracy values
    med_accuracy = df["accuracy"].median()*100
    min_accuracy = df["accuracy"].min()*100
    max_accuracy = df["accuracy"].max()*100

    # Display Results
    print("\nFinal Results:")
    print('Mean Accuracy: (%0.2f' % (mean_accuracy),u'\u00b1','%0.2f' % (std_accuracy)+')%')
    print('Median Accuracy: %0.2f' % (med_accuracy))
    print('Accuracy Range: [%0.2f, %0.2f' % (min_accuracy,max_accuracy)+']')
    print('True Positive: (%0.2f' % (mean_TP),u'\u00b1','%0.2f' % (std_TP)+')%')
    print('True Negative: (%0.2f' % (mean_TN),u'\u00b1','%0.2f' % (std_TN)+')%')
    print('False Positive: (%0.2f' % (mean_FP),u'\u00b1','%0.2f' % (std_FP)+')%')
    print('False Negative: (%0.2f' % (mean_FN),u'\u00b1','%0.2f' % (std_FN)+')%')
    print('Mean Precision: (%0.2f' % (mean_precision)+')%')
    print('Mean Recall: (%0.2f' % (mean_recall)+')%')
    print('Mean F1-Score: (%0.2f' % (mean_f1_score)+')%')
    print('Mean MCC: (%0.2f' % (mean_mcc)+')')
    #print('Avg Training Time: (%0.2f' % (mean_train_time),u'\u00b1','%0.2f' % (std_train_time)+') seconds')
    #print('Avg Test Eval Time: (%0.2f' % (mean_test_time),u'\u00b1','%0.2f' % (std_test_time)+') seconds')
        
def plot_results(results):
    '''Using Accuracy and MAE for Metrics'''
    
    
    df = pd.DataFrame(results, columns = ["accuracy","TP","TN","FP","FN","training_time","test_time"])
    
    accuracy = [element * 100 for element in df["accuracy"]]


    plt.figure()
    plt.hist(accuracy, 10)
    plt.title('Accuracy Distribution')
    plt.axis("on")
    plt.xlabel("Accuracy")
    plt.ylabel("Count")
    plt.show()
    plt.savefig('/home/Thesis/accuracy_distribution.png')

    if __name__ == '__main__': 
        main()