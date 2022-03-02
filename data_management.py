from random import shuffle
from main import * 

def get_data():
    # Example from https://www.tensorflow.org/tutorials/load_data/images
    
    '''Download the BreaKHis dataset'''

    # If data path doesn't exist, create directory to store dataset
    # and download the dataset. The data will be augmented here
    path = "/tmp/.keras/datasets"
    
    if not os.path.isdir(path):
        print('Downloading Data')
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
        print("\nDownloaded",image_count,"images\n")

    # Data is already downloaded
    else:
        data_dir = '/tmp/.keras/datasets/BreaKHis_v1/histology_slides/breast'

    return data_dir
    


def augment_data(data_dir):
    '''load and preprocess an image dataset using Keras preprocessing layers and utilities'''
    '''Create a dataset'''
    # Typically standard size for smaller data sets (~1000 samples)
    batch_size = 32 
    img_height = 700
    img_width = 460

    # Pull dataset from directory. Shuffle the dataset
    ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size,
        shuffle=True)


    # Split the data: 80% training, 10% Validation, 10% testing
    train_ds, val_ds, test_ds = get_dataset_partitions_tf(ds, len(ds))
    print('\n',len(train_ds),'\n')
    print('\n',len(test_ds),'\n')
    print('\n',len(val_ds),'\n')
    '''Configure the dataset for performance'''
    AUTOTUNE = tf.data.AUTOTUNE

    class_names = ['benign', 'malignant']
    visualize_sample_data(class_names,train_ds,'augmented_training_set')

    return train_ds, val_ds, test_ds, class_names



def get_dataset_partitions_tf(ds, ds_size, train_split=0.8, val_split=0.1, test_split=0.1, shuffle=True, shuffle_size=10000):
    assert (train_split + test_split + val_split) == 1
    
    # Shuffling here might not be necessary
    if shuffle:
        # Specify seed to always have the same split distribution between runs
        ds = ds.shuffle(shuffle_size, seed=12)
    
    # Define the size of each split based on the ds_size
    train_size = int(train_split * ds_size)
    val_size = int(val_split * ds_size)
    
    # Split the data: 80% training, 10% Validation, 10% testing
    train_ds = ds.take(train_size)    
    val_ds = ds.skip(train_size).take(val_size)
    test_ds = ds.skip(train_size).skip(val_size)
    print(test_ds)


    return train_ds, val_ds, test_ds





# def augment_data(data_dir):
#     # '''DATA AUGMENTATION'''
#     # '''https://www.tensorflow.org/tutorials/images/data_augmentation
#     #   Increases the diversity of your training set by applying random 
#     #   (but realistic) transformations, such as image rotation'''


#     # # Create a `Counter` object and `Dataset.zip` it together with the training set.
#     # counter = tf.data.experimental.Counter()
#     # train_ds = tf.data.Dataset.zip((train_ds, (counter, counter)))

#     (train_ds, val_ds, test_ds), metadata = tfds.load(
#         data_dir = data_dir,
#         # 'tf_flowers',
#         split=['train[:80%]', 'train[80%:90%]', 'train[90%:]'],
#         with_info=True,
#         as_supervised=True,
#     )

#     # Shuffle and augment the Training dataset
#     train_ds = (
#         train_ds
#         .shuffle(1000)
#         .map(support.augment, num_parallel_calls=AUTOTUNE)
#         .batch(batch_size)
#         .prefetch(AUTOTUNE)
#     )

#     # Shuffle and augment the Validation dataset
#     val_ds = (
#         val_ds
#         .map(support.resize_and_rescale, num_parallel_calls=AUTOTUNE)
#         .batch(batch_size)
#         .prefetch(AUTOTUNE)
#     )

#     # Shuffle and augment the Tesing dataset
#     test_ds = (
#         test_ds
#         .map(support.resize_and_rescale, num_parallel_calls=AUTOTUNE)
#         .batch(batch_size)
#         .prefetch(AUTOTUNE)
#     )

#     support.visualize_sample_data(class_names,train_ds,'augmented_training_set')

#     # Prints out the possible class names of the flowers
#     class_names = train_ds.class_names
#     print(class_names)

#     return train_ds,val_ds,class_names