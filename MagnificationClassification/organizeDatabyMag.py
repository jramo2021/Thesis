import os
import shutil
from pathlib import Path
import time
import tensorflow as tf

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

#download data
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
    data_dir = Path(data_dir)/'histology_slides/breast/'
    
    # Shows how many images were downloaded this time
    image_count = len(list(data_dir.glob('**/*.png')))

    end = time.time()
    print("\nDownloaded",image_count,"images in %0.2f" %(end - start),"seconds\n")


magnification = ['40X', '100X', '200X', '400X']

# Output directory where organized folders will be created
output_path = "/tmp/.keras/datasets/"

for magni in magnification:
    out = os.path.join(output_path, magni)
    os.makedirs(out, exist_ok=True)

# Iterate through each subclass in subclass path array
for subclassPath in subclassPathArray:
    # Iterate through patients 
    for dirs in os.listdir(subclassPath):
        for mag in os.listdir(os.path.join(subclassPath, dirs)):
                shutil.copytree(os.path.join(subclassPath,dirs,mag), os.path.join(output_path, mag, dirs))

print("Dataset organized by magnification successfully.")