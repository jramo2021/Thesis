# from turtle import TPen
from support_functions import *
from data_management import *
from classify import *
import pandas as pd


def main():
    
    # Start Timer
    total_start = time.time()

    '''Collect and Preprocess Data'''
    # Download Data
    data_dir = get_data()
    
    # Preprocess Data (Data Augmentation and datasplit splits)
    train_ds, val_ds, test_ds = preprocess_data(data_dir,aug_split = 0)
    
    '''Define, Train and Evaluate the Model'''
    # Test 30 times because it is a stochastic process
    results = []
    for i in range(50):
        print("\n_______Test",i+1,"_______")
        # Define and train the model 
        model, history, training_time = train(train_ds, val_ds)

        
        
    #     # Evaluate the Model (classify test dataset)
    #     results.append(model.evaluate(test_ds,batch_size=32))

    #     # Append the time it took to train the model to the results
    #     results[i].append(training_time)
    
    # # Displays true labels compared to estimated labels for 32 samples. 
    # confirm_output(model, test_ds)
    
        # Calculates TP, TN, FP, FN and total accuracy
        results.append(calculate_metrics(model,test_ds))
        results[i].append(training_time)
    
    display_results(results)

    # Plots the Loss and Accuracy over number of epochs
    # plot_history(history)
    
    
#     '''Using Accuracy and MAE for Metrics'''
#     df = pd.DataFrame(results, columns = ["loss", "accuracy","mae","time"])
    
    
#     # Calculate Mean and Standard Deviation for Performance Metrics
#     mean_loss = df["loss"].mean()
#     mean_accuracy = df["accuracy"].mean()
#     mean_mae = df["mae"].mean()
#     mean_time = df["time"].mean()
#     std_loss = df["loss"].std()
#     std_accuracy = df["accuracy"].std()
#     std_mae = df["mae"].std()
#     std_time = df["time"].std()

# # df = pd.DataFrame(results, columns = ["loss", "accuracy","mae","est","time"])

# #     # Calculate Mean and Standard Deviation for Performance Metrics
# #     mean_loss = df["loss"].mean()
# #     mean_accuracy = df["accuracy"].mean()
# #     mean_mae = df["mae"].mean()
# #     mean_est = df["est"].mean()
# #     mean_time = df["time"].mean()
# #     std_loss = df["loss"].std()
# #     std_accuracy = df["accuracy"].std()
# #     std_mae = df["mae"].std()
# #     std_est = df["est"].std()
# #     std_time = df["time"].std()

#     total_end = time.time()

#     # Display Results
#     print("\nFinal Results:")
#     print('Loss: (%0.2f' % (mean_loss*100),u'\u00b1','%0.2f' % (std_loss*100)+')%')
#     print('Accuracy: (%0.2f' % (mean_accuracy*100),u'\u00b1','%0.2f' % (std_accuracy*100)+')%')
#     # print('Estimated Accuracy: (%0.2f' % (mean_est*100),u'\u00b1','%0.2f' % (std_est*100)+')%')
#     print('Mean Absolute Error: %0.2f' % (mean_mae),u'\u00b1','%0.2f' % (std_mae))
#     print('Avg Time: (%0.2f' % (mean_time),u'\u00b1','%0.2f' % (std_time)+') seconds')
    total_end = time.time()
    print('Total Time: %0.2f' % (total_end - total_start),"seconds")

# def confirm_output(model, test_ds):
#     '''Confirms that the model predicts reasonable labels.
#     Uses the trained model to predict the labels of the first batch (32 samples) of test data.
#     An example of the predicted label will be displayed alongside the true labels for visual confirmation.
#     Since the evaluation and prediction methods have some variance, they are tested 30 times to show that 
#     they have similar accuracy distributions'''
#     mini_ds = test_ds.take(1)
#     results = []
#     for i in range(30):
#         results.append(model.evaluate(mini_ds,batch_size=32))
    
#     mini_ds, labels = next(iter(test_ds))
#     labels = tf.constant(labels).numpy()


#     prediction = model.predict(mini_ds,batch_size=32)
#     prediction_labels = prediction.argmax(axis=-1)
#     print(prediction_labels)

#     est = (1-sum(abs(prediction_labels-labels))/len(prediction_labels))
        

#     df = pd.DataFrame(results, columns = ["loss", "accuracy","mae"])
    
#     print('\nTrue Labels\n',labels)
#     print('Predicted Labels\n',prediction_labels)

#     mean_accuracy = df["accuracy"].mean()
#     std_accuracy = df["accuracy"].std()
#     print('Prediction Accuracy: %0.2f' % (est*100)+'%')
#     print('Evaluation Accuracy: (%0.2f' % (mean_accuracy*100),u'\u00b1','%0.2f' % (std_accuracy*100)+')%')

'''Calculates Accuracy of for the model classifying the test data'''
def calculate_metrics(model,test_ds):
    
    # Initialize Lists
    labels = []
    prediction_labels = []
    
    # Make test dataset iterable object
    ds = iter(test_ds)

    # For each batch, append labels to both the true label array and predicted label array
    for i in range(len(test_ds)):
        
        # Collect True Data Labels
        images, temp_labels = next(ds)
        temp_labels = tf.constant(temp_labels).numpy()
        labels.extend(temp_labels)
        
        # Collect Prediction Labels
        prediction = model.predict(images,batch_size=32)
        temp_labels = prediction.argmax(axis=-1)
        prediction_labels.extend(temp_labels)
    
    # Calculate True/False Positive/Negative
    # Initialize count
    TP = 0
    TN = 0
    FP = 0
    FN = 0

    total = len(labels)
    for i in range(total):
        if labels[i] == 1 and prediction_labels[i] == 1:
            TP += 1     # True Positive
        elif labels[i] == 0 and prediction_labels[i] == 0:
            TN += 1     # True Negative
        elif labels[i] == 0 and prediction_labels[i] == 1:
            FP += 1     # False Positive
        elif labels[i] == 1 and prediction_labels[i] == 0:
            FN += 1     # False Negative
        else:   # If labels aren't binary, raise error
            raise ValueError("Labels are not binary")

    # Calculate Accuracy and return results
    accuracy = (TN + TP)/total
    return [accuracy, TP/total, TN/total, FP/total, FN/total]
    

def display_results(results):
    '''Using Accuracy and MAE for Metrics'''
    df = pd.DataFrame(results, columns = ["accuracy","TP","TN","FP","FN","training_time"])
    
    
    # Calculate Mean for Performance Metrics
    mean_accuracy = df["accuracy"].mean()*100
    mean_TP = df["TP"].mean()*100
    mean_TN = df["TN"].mean()*100
    mean_FP = df["FP"].mean()*100
    mean_FN = df["FN"].mean()*100
    mean_time = df["training_time"].mean()

    # Calculate Standard Deviation for Performance Metrics
    std_accuracy = df["accuracy"].std()*100
    std_TP = df["TP"].std()*100
    std_TN = df["TN"].std()*100
    std_FP = df["FP"].std()*100
    std_FN = df["FN"].std()*100
    std_time = df["training_time"].std()


    # Display Results
    print("\nFinal Results:")
    print('Accuracy: (%0.2f' % (mean_accuracy),u'\u00b1','%0.2f' % (std_accuracy)+')%')
    print('True Positive: (%0.2f' % (mean_TP),u'\u00b1','%0.2f' % (std_TP)+')%')
    print('True Negative: (%0.2f' % (mean_TN),u'\u00b1','%0.2f' % (std_TN)+')%')
    print('False Positive: (%0.2f' % (mean_FP),u'\u00b1','%0.2f' % (std_FP)+')%')
    print('False Negative: (%0.2f' % (mean_FN),u'\u00b1','%0.2f' % (std_FN)+')%')
    print('Avg Training Time: (%0.2f' % (mean_time),u'\u00b1','%0.2f' % (std_time)+') seconds')
        

if __name__ == '__main__': 
    main()