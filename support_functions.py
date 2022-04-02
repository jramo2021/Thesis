import os
import matplotlib.pyplot as plt
import tensorflow as tf

''' Visualize File Structure:'''

# Prints all of the directories under the root directory
def tree_directory_printer(root):
    '''Root should be formatted with forward slashes or double back slashes'''
    for root, dirs, files in os.walk(root):
        for d in dirs:
            print (os.path.join(root, d))    

# Prints all of the files under the root directory
def tree_file_printer(root):
    '''Root should be formatted with forward slashes or double back slashes'''
    for root, dirs, files in os.walk(root):
        for f in files:
            print (os.path.join(root, f))

# tree_directory_printer('C:/Users/Ross/OneDrive/Documents/Thesis')

'''Visualizations'''
def visualize_sample_data(class_names,train_ds,filename = "visualize_sample_data"):
    # Save sample images of training data
    plt.figure(figsize=(10, 10))
    for images, labels in train_ds.take(1):
    # for images, labels in train_ds.take(2):
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(class_names[labels[i]])
            plt.axis("off")

    # Save image
    plt.savefig('/home/Thesis/'+filename+'.png')

# def visualize_augmentation(original, augmented, filename = 'augemented'):
#     fig = plt.figure()
#     plt.subplot(1,2,1)
#     plt.title('Original image')
#     plt.imshow(original)

#     plt.subplot(1,2,2)
#     plt.title('Augmented image')
#     plt.imshow(augmented)

#     # Save image
#     plt.savefig('/home/Thesis/'+ filename +'.png')


def visualize_augmentations(train_ds,filename = "augmentations"):
    # Save sample images of training data
    plt.figure(figsize=(10, 10))
    image, label = next(iter(train_ds))

    rot_and_flip_aug = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal_and_vertical",seed=123),
        tf.keras.layers.RandomRotation(factor=(-.5, .5),seed=123),
    ])

    ax = plt.subplot(3, 3, 1)
    plt.imshow(image[1].numpy().astype("uint8"))
    plt.title("Original")
    plt.axis("off")
    for i in range(8):
        # result = resize_and_rescale(image[1])
        # result = rot_and_flip_aug(result)
        result = rot_and_flip_aug(image[1])
        
        ax = plt.subplot(3, 3, i + 2)
        plt.imshow(result.numpy().astype("uint8"))
        plt.title("Possible Augment")
        plt.axis("off")

    # Save image
    plt.savefig('/home/Thesis/'+filename+'.png')

# def getSamplesFromDataGen(resultData):
#     x = resultData.next() #fetch the first batch
#     a = x[0] # train data
#     b = x[1] # train label
#     for i in range(0,5):
#         plt.imshow(a[i])
#         plt.title(b[i])
#         plt.show() 