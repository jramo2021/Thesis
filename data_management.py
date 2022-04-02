from main import * 
import os, pathlib
import tensorflow as tf
import time
import matplotlib.pyplot as plt
import copy
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

        # Define Dataset url
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

    # Data is already downloaded
    else:
        print('\nData Directory Exists\n')
        data_dir = '/tmp/.keras/datasets/BreaKHis_v1/histology_slides/breast'

    return data_dir
    
def preprocess_data(data_dir, color_mode = 'rgb'):
    '''load and preprocess an image dataset using Keras preprocessing layers and utilities'''
    '''Create a dataset'''
    # Typically standard size for smaller data sets (~1000 samples)
    start = time.time()
    batch_size = 32 
    # img_size = 512
    img_size = 256

    # Pull dataset from directory. Shuffle the dataset
    ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        # Try grayscale
        # color_mode="grayscale",
        color_mode = color_mode,
        seed=123,
        image_size=(img_size, img_size),
        # image_size=(700, 460),
        batch_size=batch_size,
        shuffle=True)

    print('\n#Training Batches:',len(ds))

    # Split the data: 80% training, 10% Validation, 10% testing
    # train_ds, val_ds, test_ds = get_dataset_partitions_tf(ds, len(ds))
    train_ds, val_ds, test_ds, augmented_data = get_dataset_partitions_tf(ds, len(ds))


    print('\n#Training Batches:',len(train_ds))
    print('\n#Valdiation Batches:',len(val_ds))
    print('\n#Testing Batches:',len(test_ds))
    
    '''Configure the dataset for performance'''
    AUTOTUNE = tf.data.AUTOTUNE

    class_names = ['benign', 'malignant']
    visualize_sample_data(class_names,train_ds,'training samples')

    # return train_ds, val_ds, test_ds, class_names
    return train_ds, val_ds, test_ds, augmented_data, class_names

# def get_dataset_partitions_tf(ds, ds_size, train_split=0.8, val_split=0.1, test_split=0.1, shuffle=True, shuffle_size=10000):
#     assert (train_split + test_split + val_split) == 1
    
#     # Shuffling here might not be necessary
#     if shuffle:
#         # Specify seed to always have the same split distribution between runs
#         ds = ds.shuffle(shuffle_size, seed=123)
    
#     total_size = train_split*(1+aug_size)+val_split + test_split
    
#     # Define the size of each split based on the ds_size
#     train_size = int(train_split * ds_size)
#     val_size = int(val_split * ds_size)
#     aug_size = int(train_split * .2 * ds_size)
    
#     # Split the data: 80% training, 10% Validation, 10% testing
#     train_ds = ds.take(train_size)
#     aug_ds = ds.take(aug_size)    
#     val_ds = ds.skip(train_size).take(val_size)
#     test_ds = ds.skip(train_size).skip(val_size)
#     print("\nTraining Set:",train_ds)
#     print("\nAugmented Set:",aug_ds)

#     return train_ds, val_ds, test_ds, aug_ds

'''Experimental: check that it works (total size is too big I think, maybe divide by total size?)'''
def get_dataset_partitions_tf(ds, ds_size, train_split=0.8, aug_split = .2, val_split=0.1, test_split=0.1, shuffle=True, shuffle_size=10000):
    assert (train_split + test_split + val_split) == 1
    
    # Shuffling here might not be necessary
    if shuffle:
        # Specify seed to always have the same split distribution between runs
        ds = ds.shuffle(shuffle_size, seed=123)    

    
    # Define the size of each split based on the ds_size and the splits
    train_size = int(train_split * ds_size)
    aug_size = int(train_split * ds_size * aug_split)

    # Expanded size accounts for adding the augmented data points
    expanded_size = 1 + train_split*aug_split
    val_size = int(val_split * expanded_size * ds_size)
    
    
    # Split the data: 80% training, 10% Validation, 10% testing
    # train_ds = ds.take(train_size)
    # aug_ds = ds.take(aug_size)    
    # val_ds = ds.skip(train_size).take(val_size)
    # test_ds = ds.skip(train_size).skip(val_size)
    
    val_ds = ds.take(val_size)
    test_ds = ds.skip(val_size).take(val_size)
    train_ds = ds.skip(val_size).skip(val_size)
    aug_ds = ds.skip(val_size).skip(val_size).take(aug_size)
    

    return train_ds, val_ds, test_ds, aug_ds

def augment_data(train_ds,aug_ds):
    '''Define Augmentations'''
    # resize_and_rescale = tf.keras.Sequential([
    #     tf.keras.layers.Resizing(700, 460),
    # ])

    # Performs random rotation and flip
    rot_and_flip_aug = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal_and_vertical",seed=123),
        tf.keras.layers.RandomRotation(factor=(-.5, .5),seed=123),
    ])

    # Applies rotation and flip to the data
    # The parallel call makes it so that it runs the rotation and 
    # flip process in parallel with mapping it to the data. Making 
    # the training take much less time (~244s -> ~112s)
    AUTOTUNE = tf.data.AUTOTUNE
    aug_ds = aug_ds.map(
        lambda x, y: (rot_and_flip_aug(x, training=True), y),num_parallel_calls=AUTOTUNE)


    images, label = next(iter(train_ds))
    print('\nBefore Concatenation')
    print("number of training batches:",len(train_ds))
    print('Batch_size:',len(images))
    print('Rows:',len(images[0]))
    print('Columns:',len(images[0][0]))
    print('RGB:',len(images[0][0][0]))

    # Append the augmented data to the training data
    train_ds = train_ds.concatenate(aug_ds)
    
    print('\nAfter Concatenation')
    images, label = next(iter(train_ds))
    print("number of training batches:",len(train_ds))
    print('Batch_size:',len(images))
    print('Rows:',len(images[0]))
    print('Columns:',len(images[0][0]))
    print('RGB:',len(images[0][0][0]))
    
    # train_ds = train_ds.shuffle(shuffle_size, seed = 123)
    
    # Show some augmentations
    visualize_augmentations(train_ds,filename = "augmentations")

    print(len(train_ds))
    return train_ds

