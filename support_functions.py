import numpy as np
import os
import PIL
import PIL.Image
import tensorflow as tf
import pathlib
import matplotlib.pyplot as plt

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

def visualize_augmentation(original, augmented, filename = 'augemented'):
    fig = plt.figure()
    plt.subplot(1,2,1)
    plt.title('Original image')
    plt.imshow(original)

    plt.subplot(1,2,2)
    plt.title('Augmented image')
    plt.imshow(augmented)

    # Save image
    plt.savefig('/home/Thesis/'+ filename +'.png')

'''Perform Augmentation: Need image size to be max(len(row),len(col))'''
def resize_and_rescale(image, label):
  image = tf.cast(image, tf.float32)
  image = tf.image.resize(image, [700, 700])
  image = (image / 255.0)
  return image, label

def augment(image_label, seed):
    image, label = image_label
    image, label = resize_and_rescale(image, label)
    # image = tf.image.resize_with_crop_or_pad(image, (700 + 6), (700 + 6))
    
    # Make a new seed.
    new_seed = tf.random.experimental.stateless_split(seed, num=1)[0, :]

    # # Random crop back to the original size.
    # image = tf.image.stateless_random_crop(
    #     image, size=[700, 700, 3], seed=seed)
    
    # Random brightness.
    image = tf.image.stateless_random_brightness(
        image, max_delta=0.5, seed=new_seed)
    image = tf.clip_by_value(image, 0, 1)
    
    return image, label