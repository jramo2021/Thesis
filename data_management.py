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

    # # # Show some example images
    # # ex_image = list(data_dir.glob('ex_image/*'))
    # # PIL.Image.open(str(ex_image[0]))
    # # PIL.Image.open(str(ex_image[1]))

    

def augment_data(data_dir):

    '''load and preprocess an image dataset using Keras preprocessing layers and utilities'''
    '''Create a dataset'''
    # Set some parameters for the loader
    batch_size = 32 # Typically standard size for smaller data sets (~1000 samples)
    img_height = 700
    img_width = 460

    # Create Training dataset: 80% of total data is for training
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)

    # Create Validation dataset: 20% of total data is for validation
    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)

    # List of possible classifications 
    class_names = train_ds.class_names

    '''Visualize the Training Data'''
    visualize_sample_data(class_names,train_ds)


    '''Standardize the Data to be within [0,1]'''
    normalization_layer = tf.keras.layers.Rescaling(1./255)

    # Can use this layer to apply it to the data set:
    # Or (as shown in flowers example) you can include the layer inside the model definition to simplify deployment
    normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    image_batch, labels_batch = next(iter(normalized_ds))


    '''Configure the dataset for performance'''
    AUTOTUNE = tf.data.AUTOTUNE

    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    return train_ds,val_ds,class_names
    


    
    # '''DATA AUGMENTATION'''
    # '''https://www.tensorflow.org/tutorials/images/data_augmentation
    #   Increases the diversity of your training set by applying random 
    #   (but realistic) transformations, such as image rotation'''


    # # # Create a `Counter` object and `Dataset.zip` it together with the training set.
    # counter = tf.data.experimental.Counter()
    # train_ds = tf.data.Dataset.zip((train_ds, (counter, counter)))

    # (train_ds, val_ds, test_ds), metadata = tfds.load(
    #     # data_dir,
    #     'tf_flowers',
    #     split=['train[:80%]', 'train[80%:90%]', 'train[90%:]'],
    #     with_info=True,
    #     as_supervised=True,
    # )

    # # Shuffle and augment the Training dataset
    # train_ds = (
    #     train_ds
    #     .shuffle(1000)
    #     .map(support.augment, num_parallel_calls=AUTOTUNE)
    #     .batch(batch_size)
    #     .prefetch(AUTOTUNE)
    # )

    # # Shuffle and augment the Validation dataset
    # val_ds = (
    #     val_ds
    #     .map(support.resize_and_rescale, num_parallel_calls=AUTOTUNE)
    #     .batch(batch_size)
    #     .prefetch(AUTOTUNE)
    # )

    # # Shuffle and augment the Tesing dataset
    # test_ds = (
    #     test_ds
    #     .map(support.resize_and_rescale, num_parallel_calls=AUTOTUNE)
    #     .batch(batch_size)
    #     .prefetch(AUTOTUNE)
    # )

    # support.visualize_sample_data(class_names,train_ds,'augmented_training_set')

    # # Prints out the possible class names of the flowers
    # class_names = train_ds.class_names
    # print(class_names)

    # return train_ds,val_ds,class_names