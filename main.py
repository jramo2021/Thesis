import numpy as np
import os
import PIL
import PIL.Image
import tensorflow as tf
import pathlib
import matplotlib.pyplot as plt
import support_functions as support
import tensorflow_datasets as tfds

from support_functions import *
from data_management import *
from classify import *

def main():
    
    # Download and Augment Data
    data_dir = get_data()
    train_ds, val_ds, class_names = augment_data(data_dir)
    
    print('\n',class_names,'\n')

    # Create and train the model using augmented data 
    model = train(train_ds, val_ds, class_names)



if __name__ == '__main__': 
    main()