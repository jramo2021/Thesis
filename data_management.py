from main import * 
import os, pathlib
import tensorflow as tf
import time
import matplotlib.pyplot as plt
import numpy as np

def get_data():
    '''If the data folder already exists, the directory for the data will be returned.
    Otherwise, the data folder will be created and BreakHis dataset will 
    be downloaded to that folder location'''
    # Example from https://www.tensorflow.org/tutorials/load_data/images
    
    '''Download the BreaKHis dataset'''

    # If data path doesn't exist, create directory to store dataset
    # and download the dataset. The data will be augmented here
    path = "/tmp/.keras/datasets"
    
    if not os.path.isdir(path):
        start = time.time()
        
        print('\n')
        
        # Create the dataset path in the Docker container
        os.makedirs(path)

        # Define Dataset url (Download Link)
        dataset_url = "http://www.inf.ufpr.br/vri/databases/BreaKHis_v1.tar.gz"

        # Download data and define directory path
        data_dir = tf.keras.utils.get_file(fname=None,
            origin=dataset_url,
            untar=True)
        data_dir = pathlib.Path(data_dir)/'histology_slides/breast/'
        
        # Shows how many images were downloaded this time
        image_count = len(list(data_dir.glob('*/*.png')))

        end = time.time()
        print("\nDownloaded",image_count,"images in %0.2f" %(end - start),"seconds\n")

    # Data is already downloaded. Define Path
    else:
        print('\nData Directory Exists\n')
        data_dir = '/tmp/.keras/datasets/BreaKHis_v1/histology_slides/breast'

    return data_dir
    
def preprocess_data(data_dir, color_mode = 'rgb',aug_split = 0):
    '''load and preprocess an image dataset using Keras preprocessing layers and utilities'''
    '''Create a dataset'''
    
    # img_size = 512
    img_size = 256
    # img_size = 224 # Needed for KGG16 and Resnet architectures

    # Pull dataset from directory. Shuffle the dataset
    ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        # Try grayscale
        # color_mode="grayscale",
        color_mode = color_mode,
        seed=123,
        image_size=(img_size, img_size),
        # image_size=(700, 460),
        batch_size=32,
        shuffle=True)
    
    

    # Split the data: 80% training, 10% Validation, 10% testing
    train_ds, val_ds, test_ds, aug_ds = get_dataset_partitions(ds,aug_split = aug_split)
    

    # If augmentation split was valid, it will perform the augmentation 
    # and append the augmented data to the training set
    print("\nPerforming Data Augmentation:",aug_ds is not None)
    if aug_ds is not None:
        train_ds = augment_data(train_ds,aug_ds)
        print('After Adding Augmentation:')
    
    print('#Total Batches:',len(train_ds) + len(val_ds) + len(test_ds))
    print('#Training Batches:',len(train_ds))
    print('#Valdiation Batches:',len(val_ds))
    print('#Testing Batches:',len(test_ds),'\n')

    
    
    '''Configure the dataset for performance'''
    AUTOTUNE = tf.data.AUTOTUNE

    class_names = ['benign', 'malignant']
    visualize_sample_data(class_names,train_ds,'training samples')

    # return train_ds, val_ds, test_ds, class_names
    return train_ds, val_ds, test_ds


def get_dataset_partitions(ds, train_split=0.8, aug_split = 0, val_split=0.1, test_split=0.1):
    
    # Check that split values sum to 1.
    assert (train_split + test_split + val_split) == 1

    # Expanded size accounts for adding the augmented data points
    expanded_size = 1 + train_split*aug_split
    val_size = int(val_split * expanded_size * len(ds))
    
    val_ds = ds.take(val_size)
    test_ds = ds.skip(val_size).take(val_size)
    train_ds = ds.skip(val_size).skip(val_size)

    print('Test_ds length',val_size)
    # Only perform augmentation if the range is valid
    if aug_split > 0 and aug_split <=1:
        # Define the size of each split based on the dataset size and the splits
        aug_size = int(train_split * len(ds) * aug_split)
        aug_ds = ds.skip(val_size).skip(val_size).take(aug_size)
    else:
        aug_ds = None
    
    return train_ds, val_ds, test_ds, aug_ds

def augment_data(train_ds,aug_ds):
    '''Define Augmentations'''


    # Performs random rotation and flip
    rot_and_flip_aug = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal_and_vertical",seed=123),
        tf.keras.layers.RandomRotation(factor=(.35,.35),seed=123),
    ])

    # Applies rotation and flip to the data
    # The parallel call makes it so that it runs the rotation and 
    # flip process in parallel with mapping it to the data. Making 
    # the training take much less time
    AUTOTUNE = tf.data.AUTOTUNE
    aug_ds = aug_ds.map(
        lambda x, y: (rot_and_flip_aug(x, training=True), y),num_parallel_calls=AUTOTUNE)


    # Append the augmented data to the training data
    train_ds = train_ds.concatenate(aug_ds)
    
    # Show some augmentations
    visualize_augmentations(train_ds,rot_and_flip_aug,filename = "augmentations")

    print(len(train_ds))
    return train_ds

