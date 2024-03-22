import tensorflow as tf
import pandas as pd
from matplotlib import pyplot as plt
import itertools
import copy
import math
import time
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

def main():
    total_start = time.time()

    #define the model
    # base_model = tf.keras.applications.vgg16.VGG16(
    #     include_top = False,
    #     weights = 'imagenet',
    #     input_shape = (224, 224, 3)
    # )

    # ResNet = tf.keras.applications.resnet50.ResNet50(
    #         include_top=False,
    #         #classes = 2,
    #         #weights = None,
    #         weights = 'imagenet',
    #         input_shape = (224, 224, 3)
    # )

    # '''Model 1.1'''
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(2)
        ])
    # for layer in ResNet.layers:
    #     layer.trainable = False

    # model = tf.keras.Sequential()

    # model.add(ResNet)
    # model.add(tf.keras.layers.Flatten())
    # model.add(tf.keras.layers.Dense(512, activation='relu'))
    # model.add(tf.keras.layers.Dense(2, activation='relu'))
    
    #compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        #loss=tf.losses.SparseCategoricalCrossentropy(from_logits=False),
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy','mae']
        )

    color_mode = 'rgb'
    img_size = 256
    #img_size = 224 #needed for vgg16 and resnet50

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

    # visualize_augmentations(train_ds, rot_flip_zoom_aug)

    # train_ds = train_ds.concatenate(tmp_ds)

    train_ds.shuffle(len(train_ds))
    test_ds.shuffle(len(test_ds))
    val_ds.shuffle(len(val_ds))

    pos, neg = count_classes(train_ds)
    total = len(train_ds)

    train_ds = train_ds.batch(32)
    test_ds = test_ds.batch(32)
    val_ds = val_ds.batch(32)

    # Compute class weights for imbalanced dataset
    weight_for_0 = (1 / neg) * (total / 2.0) #benign weighting
    weight_for_1 = (1 / pos) * (total / 2.0) #malignant weighting

    class_weights = {0: weight_for_0, 1: weight_for_1}

    print('Weight for class 0: {:.2f}'.format(weight_for_0))
    print('Weight for class 1: {:.2f}'.format(weight_for_1))
    
    #print('After Adding Augmentation:')
        
    print('#Total Batches:',len(train_ds) + len(val_ds) + len(test_ds))
    print('#Training Batches:',len(train_ds))
    print('#Valdiation Batches:',len(val_ds))
    print('#Testing Batches:',len(test_ds),'\n')


    AUTOTUNE = tf.data.AUTOTUNE

    test_ds_copy = iter(test_ds)

    # Train model many times because it is non deterministic
    results = []

    start = time.time()

    # Train the Model
    history = model.fit(
            train_ds,
            validation_data=val_ds,        
            epochs=50,
            shuffle = True,
            class_weight=class_weights
            # callbacks=[earlystopping]
            #callbacks =[model_checkpoint_callback]
        )

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
    metrics = calculate_metrics(true_labels, predicted_labels)

    # nclude training time and test prediction and evaluation times
    end = time.time()
    test_time = end - start

    # Extract accuracy, loss, validation accuracy, and validation loss from the training history
    accuracy = history.history['accuracy']
    loss = history.history['loss']
    val_accuracy = history.history['val_accuracy']
    val_loss = history.history['val_loss']

    # # Plot MAE
    # plt.figure(figsize=(8, 6))
    # plt.plot(range(1, len(mae) + 1), mae, label='MAE')
    # plt.title('Mean Absolute Error (MAE) vs. Epoch')
    # plt.xlabel('Epoch')
    # plt.ylabel('MAE')
    # plt.legend()
    # plt.grid(True)
    # plt.savefig('mae_vs_epoch.png')  # Save MAE plot as an image
    # plt.close()

    # # Plot Loss
    # plt.figure(figsize=(8, 6))
    # epochs = range(1, len(loss) + 1)
    # plt.plot(epochs, loss, 'bo-', label='Training Loss')
    # plt.plot(epochs, val_loss, 'ro-', label='Validation Loss')
    # plt.title('Training and Validation Loss')
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.legend()
    # plt.grid(True)
    # plt.savefig('loss_vs_epoch.png')  # Save Loss plot as an image
    # plt.close()
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
    
    print("Metrics:")
    print("Accuracy: %0.2f" % metrics['accuracy'])
    print("True Positives: %0.2f" % metrics['true_positives'])
    print("True Negatives: %0.2f" % metrics['true_negatives'])
    print("False Positives: %0.2f" % metrics['false_positives'])
    print("False Negatives: %0.2f" % metrics['false_negatives'])
    print("Training Time: %0.2f" % training_time)
    print('Total Time: %0.2f' % (total_end - total_start),"seconds")

def display_pred_results(results):
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

'''Collects True Class Labels and Makes Label Predictions Based on the Images'''
def predict_labels(model,test_ds):

    # Initialize Lists
    labels = []
    pred_labels = []

    # For each batch, append labels to both the true label array and predicted label array
    for images, temp_labels in test_ds:
        
        # Collect True Data Labels
        temp_labels = tf.constant(temp_labels).numpy()
        labels.extend(temp_labels)

        # Collect Prediction Labels
        prediction = model.predict(images,batch_size=32)
        temp_pred_labels = prediction.argmax(axis=-1)
        pred_labels.extend(temp_pred_labels)

    # print(temp_labels[:6])
    # print(prediction[:6])
    # print(temp_pred_labels[:6])
    num = 0
    for i in range(len(labels)):
        num += abs(pred_labels[i]-labels[i])
    print("accuracy:\n",(1-num/len(pred_labels)))

    return labels, pred_labels

def calculate_metrics(true_labels, predicted_labels):
    # Initialize variables for metrics
    true_positives = 0
    true_negatives = 0
    false_positives = 0
    false_negatives = 0

    # Calculate metrics
    for true, pred in zip(true_labels, predicted_labels):
        if true == 1 and pred == 1:
            true_positives += 1
        elif true == 0 and pred == 0:
            true_negatives += 1
        elif true == 0 and pred == 1:
            false_positives += 1
        elif true == 1 and pred == 0:
            false_negatives += 1

    # Calculate accuracy
    total_samples = len(true_labels)
    accuracy = (true_positives + true_negatives) / total_samples if total_samples > 0 else 0.0

    # Return the calculated metrics
    return {
        'accuracy': accuracy * 100,
        'true_positives': (true_positives/total_samples) * 100,
        'true_negatives': (true_negatives/total_samples) * 100,
        'false_positives': (false_positives/total_samples) * 100,
        'false_negatives': (false_negatives/total_samples) * 100
    }
    
def display_eval_results(results):
    '''Using Accuracy and MAE for Metrics'''
    df = pd.DataFrame(results, columns = ["loss", "accuracy","mae","time"])
    
    # Calculate Mean and Standard Deviation for Performance Metrics
    mean_loss = df["loss"].mean()
    mean_accuracy = df["accuracy"].mean()
    mean_mae = df["mae"].mean()
    mean_time = df["time"].mean()
    std_loss = df["loss"].std()
    std_accuracy = df["accuracy"].std()
    std_mae = df["mae"].std()
    std_time = df["time"].std()

    # Display Results
    print("\nFinal Results:")
    print('Loss: (%0.2f' % (mean_loss*100),u'\u00b1','%0.2f' % (std_loss*100)+')%')
    print('Accuracy: (%0.2f' % (mean_accuracy*100),u'\u00b1','%0.2f' % (std_accuracy*100)+')%')
    print('Mean Absolute Error: %0.2f' % (mean_mae),u'\u00b1','%0.2f' % (std_mae))
    print('Avg Time: (%0.2f' % (mean_time),u'\u00b1','%0.2f' % (std_time)+') seconds')

def count_classes(dataset):
    positive_count = 0
    negative_count = 0

    for _, label in dataset:
        if label == 1:
            positive_count += 1
        elif label == 0:
            negative_count += 1

    return positive_count, negative_count

if __name__ == '__main__': 
    main()