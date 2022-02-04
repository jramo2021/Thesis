import numpy as np
import os
import PIL
import PIL.Image
import tensorflow as tf
import pathlib
import matplotlib.pyplot as plt
import support_functions as support

# Example from https://www.tensorflow.org/tutorials/load_data/images

'''Download the BreaKHis dataset'''

# Create directory to store dataset
path = "tmp/.keras/datasets"
if not os.path.isdir(path):
  os.makedirs("tmp/.keras/datasets")

# Define Dataset url
dataset_url = "http://www.inf.ufpr.br/vri/databases/BreaKHis_v1.tar.gz"
# dataset_url = "https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz"

# Download data and define directory path
data_dir = tf.keras.utils.get_file(fname=None,
  origin=dataset_url,
  untar=True)
data_dir = pathlib.Path(data_dir)/'histology_slides/breast/'

# # [Optional] Print the directories in the dataset folder 
# support.tree_directory_printer('/tmp/.keras/datasets')
support.tree_file_printer('/tmp/.keras/datasets')
# print("\n")


# Should be a lot images
image_count = len(list(data_dir.glob('*/*.png')))
print("Downloaded",image_count,"images")

# # # Show some example images
# # ex_image = list(data_dir.glob('ex_image/*'))
# # PIL.Image.open(str(ex_image[0]))
# # PIL.Image.open(str(ex_image[1]))

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

# Prints out the possible class names of the flowers
class_names = train_ds.class_names
print(class_names)

'''Visualize the Data'''
plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
# for images, labels in train_ds.take(2):
  for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(class_names[labels[i]])
    plt.axis("off")

# plt.savefig('home/Thesis/test_image'+str(i)+'.jpg')
plt.savefig('/home/Thesis/visualize_data.png')
# for image_batch, labels_batch in train_ds:
#   print(image_batch.shape)
#   print(labels_batch.shape)
#   break
print(data_dir)