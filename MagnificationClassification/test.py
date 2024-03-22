import os
import shutil
from pathlib import Path
import time
import tensorflow as tf
import cv2
img = cv2.imread('//tmp/.keras/datasets/BreaKHis_v1/histology_slides/breast/benign/SOB/fibroadenoma/SOB_B_F_14-14134/40X/SOB_B_F-14-14134-40-001.png')
cv2.imwrite('//home/Thesis/Vgg16Resnet50/original.png', img)

# average_blue = cv2.mean(img)[0]
# average_green = cv2.mean(img)[1]
# average_red = cv2.mean(img)[2]
# print(average_blue)
# print(average_green)
# print(average_red)

# path = "/tmp/.keras/datasets"
    
# if not os.path.isdir(path):
#     start = time.time()
    
#     print('\n')
    
#     # Create the dataset path in the Docker container
#     os.makedirs(path)

#     # Define Dataset url (Download Link)
#     dataset_url = "http://www.inf.ufpr.br/vri/databases/BreaKHis_v1.tar.gz"

#     # Download data and define directory path
#     data_dir = tf.keras.utils.get_file(fname=None,
#         origin=dataset_url,
#         untar=True)
#     data_dir = Path(data_dir)/'histology_slides/breast/'
    
#     # Shows how many images were downloaded this time
#     image_count = len(list(data_dir.glob('**/*.png')))

#     end = time.time()
#     print("\nDownloaded",image_count,"images in %0.2f" %(end - start),"seconds\n")