from main import * 
import os, pathlib
import tensorflow as tf
import time
import matplotlib.pyplot as plt
import numpy as np
import random
import shutil

subclassPathArray = [
    "/tmp/.keras/datasets/BreaKHis_v1/histology_slides/breast/benign/SOB/adenosis/",
    "/tmp/.keras/datasets/BreaKHis_v1/histology_slides/breast/benign/SOB/fibroadenoma/",
    "/tmp/.keras/datasets/BreaKHis_v1/histology_slides/breast/benign/SOB/phyllodes_tumor/",
    "/tmp/.keras/datasets/BreaKHis_v1/histology_slides/breast/benign/SOB/tubular_adenoma/",
    "/tmp/.keras/datasets/BreaKHis_v1/histology_slides/breast/malignant/SOB/ductal_carcinoma/",
    "/tmp/.keras/datasets/BreaKHis_v1/histology_slides/breast/malignant/SOB/lobular_carcinoma/",
    "/tmp/.keras/datasets/BreaKHis_v1/histology_slides/breast/malignant/SOB/mucinous_carcinoma/",
    "/tmp/.keras/datasets/BreaKHis_v1/histology_slides/breast/malignant/SOB/papillary_carcinoma/"
]

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
        image_count = len(list(data_dir.glob('**/*.png')))

        end = time.time()
        print("\nDownloaded",image_count,"images in %0.2f" %(end - start),"seconds\n")

    # Data is already downloaded. Define Path
    else:
        print('\nData Directory Exists\n')
        data_dir = '/tmp/.keras/datasets/BreaKHis_v1/histology_slides/breast'

    return data_dir
    

def organizePatientPartition():
    random.seed(321)
    destPath = "/tmp/.keras/TrainTestValData"
    if not os.path.isdir(destPath):
        os.mkdir(destPath)
        os.makedirs(destPath + '/Train/benign/')
        os.makedirs(destPath + '/Train/malignant/')
        os.makedirs(destPath + '/Test/benign/')
        os.makedirs(destPath + '/Test/malignant/')
        os.makedirs(destPath + '/Val/benign/')
        os.makedirs(destPath + '/Val/malignant/')


    percentageinTestandVal = 0.1
    for subclass in subclassPathArray:
        count = 0
        if 'adenosis' in subclass:
            patients = [name for name in os.listdir(subclass)]
            order = random.sample(range(len(patients)), k=len(patients))
            for i in order:
                if count < len(patients) - 2 * (int(percentageinTestandVal*len(patients)) + 1):
                    shutil.copytree(subclass + patients[i], destPath + '/Train/benign/' + patients[i])
                    count += 1
                elif count >= (len(patients) - 2 * (int(percentageinTestandVal*len(patients)) + 1)) and (count < len(patients) - (int(percentageinTestandVal*len(patients)) + 1)):
                    shutil.copytree(subclass + patients[i], destPath + '/Test/benign/' + patients[i])
                    count += 1
                else:
                    shutil.copytree(subclass + patients[i], destPath + '/Val/benign/' + patients[i])
                    count += 1
        elif 'fibroadenoma' in subclass:
            patients = [name for name in os.listdir(subclass)]
            order = random.sample(range(len(patients)), k=len(patients))
            for i in order:
                if count < len(patients) - 2 * (int(percentageinTestandVal*len(patients)) + 1):
                    shutil.copytree(subclass + patients[i], destPath + '/Train/benign/' + patients[i])
                    count += 1
                elif count >= (len(patients) - 2 * (int(percentageinTestandVal*len(patients)) + 1)) and (count < len(patients) - (int(percentageinTestandVal*len(patients)) + 1)):
                    shutil.copytree(subclass + patients[i], destPath + '/Test/benign/' + patients[i])
                    count += 1
                else:
                    shutil.copytree(subclass + patients[i], destPath + '/Val/benign/' + patients[i])
                    count += 1
        elif 'phyllodes' in subclass:
            patients = [name for name in os.listdir(subclass)]
            order = random.sample(range(len(patients)), k=len(patients))
            for i in order:
                if count < len(patients) - 2 * (int(percentageinTestandVal*len(patients)) + 1):
                    shutil.copytree(subclass + patients[i], destPath + '/Train/benign/' + patients[i])
                    count += 1
                elif count >= (len(patients) - 2 * (int(percentageinTestandVal*len(patients)) + 1)) and (count < len(patients) - (int(percentageinTestandVal*len(patients)) + 1)):
                    shutil.copytree(subclass + patients[i], destPath + '/Test/benign/' + patients[i])
                    count += 1
                else:
                    shutil.copytree(subclass + patients[i], destPath + '/Val/benign/' + patients[i])
                    count += 1
        elif 'tubular' in subclass:
            patients = [name for name in os.listdir(subclass)]
            order = random.sample(range(len(patients)), k=len(patients))
            for i in order:
                if count < len(patients) - 2 * (int(percentageinTestandVal*len(patients)) + 1):
                    shutil.copytree(subclass + patients[i], destPath + '/Train/benign/' + patients[i])
                    count += 1
                elif count >= (len(patients) - 2 * (int(percentageinTestandVal*len(patients)) + 1)) and (count < len(patients) - (int(percentageinTestandVal*len(patients)) + 1)):
                    shutil.copytree(subclass + patients[i], destPath + '/Test/benign/' + patients[i])
                    count += 1
                else:
                    shutil.copytree(subclass + patients[i], destPath + '/Val/benign/' + patients[i])
                    count += 1
        elif 'ductal' in subclass:
            patients = [name for name in os.listdir(subclass)]
            order = random.sample(range(len(patients)), k=len(patients))
            for i in order:
                if count < len(patients) - 2 * (int(percentageinTestandVal*len(patients)) + 1):
                    shutil.copytree(subclass + patients[i], destPath + '/Train/malignant/' + patients[i])
                    count += 1
                elif count >= (len(patients) - 2 * (int(percentageinTestandVal*len(patients)) + 1)) and (count < len(patients) - (int(percentageinTestandVal*len(patients)) + 1)):
                    shutil.copytree(subclass + patients[i], destPath + '/Test/malignant/' + patients[i])
                    count += 1
                else:
                    shutil.copytree(subclass + patients[i], destPath + '/Val/malignant/' + patients[i])
                    count += 1
        elif 'lobular' in subclass:
            patients = [name for name in os.listdir(subclass)]
            order = random.sample(range(len(patients)), k=len(patients))
            for i in order:
                if count < len(patients) - 2 * (int(percentageinTestandVal*len(patients)) + 1):
                    shutil.copytree(subclass + patients[i], destPath + '/Train/malignant/' + patients[i])
                    count += 1
                elif count >= (len(patients) - 2 * (int(percentageinTestandVal*len(patients)) + 1)) and (count < len(patients) - (int(percentageinTestandVal*len(patients)) + 1)):
                    shutil.copytree(subclass + patients[i], destPath + '/Test/malignant/' + patients[i])
                    count += 1
                else:
                    shutil.copytree(subclass + patients[i], destPath + '/Val/malignant/' + patients[i])
                    count += 1
        elif 'mucinous' in subclass:
            patients = [name for name in os.listdir(subclass)]
            order = random.sample(range(len(patients)), k=len(patients))
            for i in order:
                if count < len(patients) - 2 * (int(percentageinTestandVal*len(patients)) + 1):
                    shutil.copytree(subclass + patients[i], destPath + '/Train/malignant/' + patients[i])
                    count += 1
                elif count >= (len(patients) - 2 * (int(percentageinTestandVal*len(patients)) + 1)) and (count < len(patients) - (int(percentageinTestandVal*len(patients)) + 1)):
                    shutil.copytree(subclass + patients[i], destPath + '/Test/malignant/' + patients[i])
                    count += 1
                else:
                    shutil.copytree(subclass + patients[i], destPath + '/Val/malignant/' + patients[i])
                    count += 1
        elif 'papillary' in subclass:
            patients = [name for name in os.listdir(subclass)]
            order = random.sample(range(len(patients)), k=len(patients))
            for i in order:
                if count < len(patients) - 2 * (int(percentageinTestandVal*len(patients)) + 1):
                    shutil.copytree(subclass + patients[i], destPath + '/Train/malignant/' + patients[i])
                    count += 1
                elif count >= (len(patients) - 2 * (int(percentageinTestandVal*len(patients)) + 1)) and (count < len(patients) - (int(percentageinTestandVal*len(patients)) + 1)):
                    shutil.copytree(subclass + patients[i], destPath + '/Test/malignant/' + patients[i])
                    count += 1
                else:
                    shutil.copytree(subclass + patients[i], destPath + '/Val/malignant/' + patients[i])
                    count += 1

def preprocess_data(data_dir, color_mode = 'rgb',aug_split = 0, split_by_patient = False, balance_dataset = False):
    '''load and preprocess an image dataset using Keras preprocessing layers and utilities'''
    '''Create a dataset'''

    #img_size = 512
    img_size = 256
    # img_size = 224 # Needed for KGG16 and Resnet architectures
    if split_by_patient & ~balance_dataset:
        if not os.path.isdir('/tmp/.keras/TrainTestValData'):
            organizePatientPartition()

        train_ds = tf.keras.utils.image_dataset_from_directory(
            '/tmp/.keras/TrainTestValData/Train',
            # Try grayscale
            # color_mode="grayscale",
            color_mode = color_mode,
            seed=123,
            image_size=(img_size, img_size),
            # image_size=(700, 460),
            batch_size=32,
            shuffle=True)
    
        test_ds = tf.keras.utils.image_dataset_from_directory(
            '/tmp/.keras/TrainTestValData/Test',
            # Try grayscale
            # color_mode="grayscale",
            color_mode = color_mode,
            seed=123,
            image_size=(img_size, img_size),
            # image_size=(700, 460),
            batch_size=32,
            shuffle=True)

        val_ds = tf.keras.utils.image_dataset_from_directory(
            '/tmp/.keras/TrainTestValData/Val',
            # Try grayscale
            # color_mode="grayscale",
            color_mode = color_mode,
            seed=123,
            image_size=(img_size, img_size),
            # image_size=(700, 460),
            batch_size=32,
            shuffle=True)

        # augments data if uncommented
        rot_flip_zoom_aug = tf.keras.Sequential([
            tf.keras.layers.RandomFlip("horizontal",seed=123),
            tf.keras.layers.RandomRotation(factor=0.35,seed=123),
            tf.keras.layers.RandomZoom(height_factor = (-0.3, 0), seed=123)
            ])
        
        tmp_ds = train_ds.map(
        lambda x, y: (rot_flip_zoom_aug(x, training=True), y),num_parallel_calls=tf.data.AUTOTUNE)

        #visualize_augmentations(train_ds, rot_flip_zoom_aug)

        train_ds = train_ds.concatenate(tmp_ds)
        train_ds = train_ds.shuffle(len(train_ds))

    elif balance_dataset & ~split_by_patient:
        #creates a balanced data set by taking 1000 images from each subclass at random 
        #and applying data augmentation 
        #FIXED FIX IMAGES GETTING REWRITTEN BY DIFFERENT SUBCLASSES
        
        train_A = tf.keras.utils.image_dataset_from_directory(
            '/tmp/.keras/BalancedDataset/Adenosis',
            # Try grayscale
            # color_mode="grayscale",
            color_mode = color_mode,
            seed=123,
            image_size=(img_size, img_size),
            # image_size=(700, 460),
            batch_size=None,
            shuffle=True)

        train_F = tf.keras.utils.image_dataset_from_directory(
            '/tmp/.keras/BalancedDataset/Fibroadenoma',
            # Try grayscale
            # color_mode="grayscale",
            color_mode = color_mode,
            seed=123,
            image_size=(img_size, img_size),
            # image_size=(700, 460),
            batch_size=None,
            shuffle=True)

        train_PT = tf.keras.utils.image_dataset_from_directory(
            '/tmp/.keras/BalancedDataset/Phyllodes',
            # Try grayscale
            # color_mode="grayscale",
            color_mode = color_mode,
            seed=123,
            image_size=(img_size, img_size),
            # image_size=(700, 460),
            batch_size=None,
            shuffle=True)

        train_T = tf.keras.utils.image_dataset_from_directory(
            '/tmp/.keras/BalancedDataset/Tubular',
            # Try grayscale
            # color_mode="grayscale",
            color_mode = color_mode,
            seed=123,
            image_size=(img_size, img_size),
            # image_size=(700, 460),
            batch_size=None,
            shuffle=True)

        train_D = tf.keras.utils.image_dataset_from_directory(
            '/tmp/.keras/BalancedDataset/Ductal',
            # Try grayscale
            # color_mode="grayscale",
            color_mode = color_mode,
            seed=123,
            image_size=(img_size, img_size),
            # image_size=(700, 460),
            batch_size=None,
            shuffle=True)
        
        train_L = tf.keras.utils.image_dataset_from_directory(
            '/tmp/.keras/BalancedDataset/Lobular',
            # Try grayscale
            # color_mode="grayscale",
            color_mode = color_mode,
            seed=123,
            image_size=(img_size, img_size),
            # image_size=(700, 460),
            batch_size=None,
            shuffle=True)

        train_M = tf.keras.utils.image_dataset_from_directory(
            '/tmp/.keras/BalancedDataset/Mucinous',
            # Try grayscale
            # color_mode="grayscale",
            color_mode = color_mode,
            seed=123,
            image_size=(img_size, img_size),
            # image_size=(700, 460),
            batch_size=None,
            shuffle=True)

        train_P = tf.keras.utils.image_dataset_from_directory(
            '/tmp/.keras/BalancedDataset/Papillary',
            # Try grayscale
            # color_mode="grayscale",
            color_mode = color_mode,
            seed=123,
            image_size=(img_size, img_size),
            # image_size=(700, 460),
            batch_size=None,
            shuffle=True)

        test_ds = tf.keras.utils.image_dataset_from_directory(
            '/tmp/.keras/BalancedDataset/Test',
            # Try grayscale
            # color_mode="grayscale",
            color_mode = color_mode,
            seed=123,
            image_size=(img_size, img_size),
            # image_size=(700, 460),
            batch_size=32,
            shuffle=True)

        val_ds = tf.keras.utils.image_dataset_from_directory(
            '/tmp/.keras/BalancedDataset/Val',
            # Try grayscale
            # color_mode="grayscale",
            color_mode = color_mode,
            seed=123,
            image_size=(img_size, img_size),
            # image_size=(700, 460),
            batch_size=32,
            shuffle=True)

        AUTOTUNE = tf.data.AUTOTUNE

        rot_flip_zoom_aug = tf.keras.Sequential([
            tf.keras.layers.RandomFlip("horizontal",seed=123),
            tf.keras.layers.RandomRotation(factor=0.35,seed=123),
            tf.keras.layers.RandomZoom(height_factor = (-0.3, 0), seed=123)
            ])

        tmp_ds = train_A.map(
        lambda x, y: (rot_flip_zoom_aug(x, training=True), y),num_parallel_calls=AUTOTUNE)
        train_A = train_A.concatenate(tmp_ds)

        tmp_ds = train_F.map(
        lambda x, y: (rot_flip_zoom_aug(x, training=True), y),num_parallel_calls=AUTOTUNE)
        train_F = train_F.concatenate(tmp_ds)

        tmp_ds = train_PT.map(
        lambda x, y: (rot_flip_zoom_aug(x, training=True), y),num_parallel_calls=AUTOTUNE)
        train_PT = train_PT.concatenate(tmp_ds)

        tmp_ds = train_T.map(
        lambda x, y: (rot_flip_zoom_aug(x, training=True), y),num_parallel_calls=AUTOTUNE)
        train_T = train_T.concatenate(tmp_ds)

        train_ds = train_A.concatenate(train_F).concatenate(train_PT).concatenate(train_T).concatenate(train_P).concatenate(train_M).concatenate(train_L).concatenate(train_D)

        train_ds = train_ds.shuffle(len(train_ds))
        train_ds = train_ds.batch(32)


        #do not remove this line or server will crash
        shutil.rmtree("/tmp/.keras/datasets/")

        AUTOTUNE = tf.data.AUTOTUNE
    
    else:
        ds = tf.keras.utils.image_dataset_from_directory(
            '/tmp/.keras/datasets/BreaKHis_v1/histology_slides/breast',
            # Try grayscale
            # color_mode="grayscale",
            color_mode = color_mode,
            seed=123,
            image_size=(img_size, img_size),
            # image_size=(700, 460),
            batch_size=32,
            shuffle=True)

        test_ds = tf.keras.utils.image_dataset_from_directory(
            '/tmp/.keras/TrainTestValData/Test',
            # Try grayscale
            # color_mode="grayscale",
            color_mode = color_mode,
            seed=123,
            image_size=(img_size, img_size),
            # image_size=(700, 460),
            batch_size=32,
            shuffle=True)
        
        # AUTOTUNE = tf.data.AUTOTUNE

        # rot_flip_zoom_aug = tf.keras.Sequential([
        #     tf.keras.layers.RandomFlip("horizontal",seed=123),
        #     tf.keras.layers.RandomRotation(factor=0.35,seed=123),
        #     tf.keras.layers.RandomZoom(height_factor = (-0.3, 0), seed=123)
        #     ])

        # aug_ds = ds.map(
        # lambda x, y: (rot_flip_zoom_aug(x, training=True), y),num_parallel_calls=AUTOTUNE)
        
        # visualize_augmentations(ds,rot_flip_zoom_aug, filename = "augmentations")

        # ds = ds.concatenate(aug_ds)

        # ds.shuffle(len(ds))

        val_size = int(0.1 * len(ds))
        val_ds = ds.take(val_size)
        #test_ds = ds.skip(val_size).take(val_size)
        train_ds = ds.skip(val_size)#.skip(val_size)

    # If augmentation split was valid, it will perform the augmentation 
    # and append the augmented data to the training set
    # print("\nPerforming Data Augmentation:",aug_ds is not None)
    # if aug_ds is not None:
    #     #train_ds = augment_data(train_ds,train_ds)
    #     #val_ds = augment_data(val_ds, val_ds)
    #     #test_ds = augment_data(test_ds, test_ds)
    #     print('After Adding Augmentation:')
    
    print('#Total Batches:',len(train_ds) + len(val_ds) + len(test_ds))
    print('#Training Batches:',len(train_ds))
    print('#Valdiation Batches:',len(val_ds))
    print('#Testing Batches:',len(test_ds),'\n')

    
    
    '''Configure the dataset for performance'''
    AUTOTUNE = tf.data.AUTOTUNE

    class_names = ['benign', 'malignant']
    #visualize_sample_data(class_names,train_ds,'training samples')

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
    #visualize_augmentations(train_ds,rot_and_flip_aug,filename = "augmentations")

    return train_ds
