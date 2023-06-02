from support_functions import *
from data_management import *
from classify import *
import pandas as pd
from matplotlib import pyplot as plt
import itertools
import copy
import math


def main():
    
    tf.debugging.set_log_device_placement(True)
    # Start Timer
    total_start = time.time()

    '''Collect and Preprocess Data'''
    # Download Data and return directory that contains the data
    data_dir = get_data()
    
    # Preprocess Data (Data Augmentation and datasplit splits)
    train_ds, val_ds, test_ds = preprocess_data(data_dir,aug_split = 0,split_by_patient = True,balance_dataset=False)
    #display_image(['Benign','Malignant'],train_ds,filename = "700x460")
    # Make the test_dataset an iterable object
    test_ds_copy = iter(test_ds)

    # Train model many times because it is non deterministic
    results = []
    best_accuracy = 0
    num_iterations = 50
    for i in range(num_iterations):
        print("\n_______Test",i+1,"_______")
        
        # Create copy of the datasets so that it can be reused
        test_ds, test_ds_copy = itertools.tee(test_ds_copy)

        # Define the Model
        model = define_model()

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
    
    #     # Evaluate the Model (classify test dataset)
    #     results.append(model.evaluate(test_ds,batch_size=32))
    #     results[i].append(training_time)
        

    # display_eval_results(results)

    # # Displays true labels compared to estimated labels for 32 samples. (Proof of Concept)
    # confirm_output(model, test_ds)
    
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

    # Return the best model
    return best_model 

def confirm_output(model, test_ds):
    '''Confirms that the model predicts reasonable labels.
    Uses the trained model to predict the labels of the test data.
    The predicted label will be displayed alongside the true labels for visual confirmation.
    Since the evaluation and prediction methods have some variance, they are tested 30 times to show that 
    they have similar accuracy distributions'''
    
    # Make test dataset an iterable object and create a back up test dataset
    test_ds, test_ds_copy = itertools.tee(iter(test_ds.skip(2).take(1)))
    
    # Use Keras evaluate method to get results
    results = model.evaluate(test_ds,batch_size=32)
    
    # Use Keras prediction method for each batch and calculate results. 
    # Uses the copy dataset so that the batches are the same as the original
    total_errors = 0 
    length = 0
    for mini_ds, labels in test_ds_copy:
        
        # Obtain true labels for batch
        labels = tf.constant(labels).numpy()
        
        # Obtain predicted labels for batch
        prediction = model.predict(mini_ds,batch_size=32)
        predicted_labels = prediction.argmax(axis=-1)
        
        # Accumulate total error count    
        total_errors += sum(abs(predicted_labels-labels))
        length += len(labels)
    
    # Calculate prediction error
    pred_acc = (1-total_errors/length)

    # Display some Labels for visual comparison
    print("\nTrue Labels:\n",labels)
    print("Predicted Labels:\n",predicted_labels)

    # Display results
    mean_acc = results[1]
    print('\nPrediction Accuracy: %0.2f' % (pred_acc*100)+'%')
    print('Evaluation Accuracy: %0.2f' % (mean_acc*100)+'%')


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

def calculate_metrics(labels, pred_labels):
    # Calculate True/False Positive/Negative
    # Initialize count
    TP = 0
    TN = 0
    FP = 0
    FN = 0

    total = len(labels)
    for i in range(total):
        if labels[i] == 1 and pred_labels[i] == 1:
            TP += 1     # True Positive
        elif labels[i] == 0 and pred_labels[i] == 0:
            TN += 1     # True Negative
        elif labels[i] == 0 and pred_labels[i] == 1:
            FP += 1     # False Positive
        elif labels[i] == 1 and pred_labels[i] == 0:
            FN += 1     # False Negative
        # else:   # If labels aren't binary, raise error
        #     raise ValueError("Labels are not binary")

    # Calculate Accuracy and return results
    accuracy = (TN + TP)/total
    return [accuracy, TP/total, TN/total, FP/total, FN/total]
    
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

def display_pred_results(results):
    '''Using Accuracy and MAE for Metrics'''
    df = pd.DataFrame(results, columns = ["accuracy","TP","TN","FP","FN","training_time","test_time"])
    
    # Calculate Mean for Performance Metrics
    mean_accuracy = df["accuracy"].mean()*100
    mean_TP = df["TP"].mean()*100
    mean_TN = df["TN"].mean()*100
    mean_FP = df["FP"].mean()*100
    mean_FN = df["FN"].mean()*100
    mean_train_time = df["training_time"].mean()
    mean_test_time = df["test_time"].mean()
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
    std_train_time = df["training_time"].std()
    std_test_time = df["test_time"].std()

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
    print('Avg Training Time: (%0.2f' % (mean_train_time),u'\u00b1','%0.2f' % (std_train_time)+') seconds')
    print('Avg Test Eval Time: (%0.2f' % (mean_test_time),u'\u00b1','%0.2f' % (std_test_time)+') seconds')
        
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