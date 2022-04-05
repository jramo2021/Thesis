from support_functions import *
from data_management import *
from classify import *
import pandas as pd


def main():
    total_start = time.time()
    # Download and Augment Data
    data_dir = get_data()
    train_ds, val_ds, test_ds = preprocess_data(data_dir)
    results = []
    print("\nEvaluating Model")
    
    total_ds = len(train_ds) + len(val_ds) + len(test_ds)
    print("train size:",len(train_ds))
    print("val size:",len(val_ds))
    print("test size:",len(test_ds))
    print("total size:",total_ds)

    # Test 30 times because it is a stochastic process
    # for i in range(0,30):
    for i in range(1):
        print("\n_______Test",i,"_______")
        start = time.time()
        
        # Create and train the model using augmented data 
        model, history = train(train_ds, val_ds)
        
        results.append(model.evaluate(test_ds,batch_size=32))
        # results.append(model.predict(test_ds,batch_size=32))
        end = time.time()
        results[i].append(end-start)
        
    print(results)
    print('\n',history.history['loss'])
    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
    plt.savefig('/home/Thesis/Loss Plot.png')
    
    plt.figure()
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
    plt.savefig('/home/Thesis/Accuracy Plot.png')
    
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

    total_end = time.time()

    # Display Results
    print("\nFinal Results:")
    print('Loss: (%0.2f' % (mean_loss*100),u'\u00b1','%0.2f' % (std_loss*100)+')%')
    print('Accuracy: (%0.2f' % (mean_accuracy*100),u'\u00b1','%0.2f' % (std_accuracy*100)+')%')
    print('Mean Absolute Error: %0.2f' % (mean_mae),u'\u00b1','%0.2f' % (std_mae))
    print('Avg Time: (%0.2f' % (mean_time),u'\u00b1','%0.2f' % (std_time)+') seconds')
    print('Total Time: %0.2f' % (total_end - total_start),"seconds")

    

if __name__ == '__main__': 
    main()