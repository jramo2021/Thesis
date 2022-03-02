import numpy as np
import sys
import os
import PIL
import PIL.Image
import tensorflow as tf
import pathlib
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds

# sys.path.append("/home/Thesis/")
# import my_dataset
from support_functions import *
from data_management import *
from classify import *

def main():
    
    # Download and Augment Data
    data_dir = get_data()
    # train_ds, val_ds, test_ds, class_names = augment_data(data_dir)
    # train_ds, val_ds, class_names = augment_data(data_dir)
    train_ds, val_ds, test_ds, class_names = augment_data(data_dir)
    print('\n',class_names,'\n')

    # Create and train the model using augmented data 
    model = train(train_ds, val_ds)

    print("\nEvaluating Model")
    results = model.evaluate(test_ds,batch_size=32)
    # results = model.evaluate(val_ds,batch_size=32)
    print("\nFinal Results:", results)

if __name__ == '__main__': 
    main()