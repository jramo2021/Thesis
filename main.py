from support_functions import *
from data_management import *
from classify import *
import pandas as pd


def main():
    total_start = time.time()
    # Download and Augment Data
    data_dir = get_data()
    train_ds, val_ds, test_ds, augmented_data, class_names = preprocess_data(data_dir)
    train_ds = augment_data(train_ds,augmented_data)
    results = []
    print("\nEvaluating Model")
    
    # Test 100 times because it is a stochastic process
    # for i in range(0,30):
    for i in range(1):
        start = time.time()
        
        # Create and train the model using augmented data 
        model = train(train_ds, val_ds)
        
        results.append(model.evaluate(test_ds,batch_size=32))
        # results.append(model.predict(test_ds,batch_size=32))
        end = time.time()
        results[i].append(end-start)
    df = pd.DataFrame(results, columns = ["loss", "sparse_categorical_accuracy","time"])
    # # df = pd.DataFrame(results, columns = ["loss", "sparse_categorical_accuracy"])
    # # print(results)
    # # print(len(results))
    mean_loss = df["loss"].mean()
    mean_accuracy = df["sparse_categorical_accuracy"].mean()
    mean_time = df["time"].mean()
    std_loss = df["loss"].std()
    std_accuracy = df["sparse_categorical_accuracy"].std()
    std_time = df["time"].std()

    total_end = time.time()

    print("\nFinal Results:")
    print('Loss: (%0.2f' % (mean_loss*100),u'\u00b1','%0.2f' % (std_loss*100)+')%')
    print('Accuracy: (%0.2f' % (mean_accuracy*100),u'\u00b1','%0.2f' % (std_accuracy*100)+')%')
    print('Avg Time: (%0.2f' % (mean_time),u'\u00b1','%0.2f' % (std_time)+') seconds')
    print('Total Time: %0.2f' % (total_end - total_start),"seconds")


if __name__ == '__main__': 
    main()