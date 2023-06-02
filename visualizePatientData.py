import os
import matplotlib.pyplot as plt
import matplotlib.image as img
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


subclass = subclassPathArray[0]
patients = [name for name in os.listdir(subclass)]
magnification = '40X'

path = subclass+patients[0]+'/'+magnification
images = [name for name in os.listdir(path)]
images.sort()
image_name1 = images[8]

image1 = img.imread(path+'/'+image_name1)
ax = plt.subplot(2, 1, 1)
plt.imshow(image1)
plt.title(image_name1)
plt.axis("off")

path = subclass+patients[1]+'/'+magnification
images = [name for name in os.listdir(path)]
images.sort()
image_name2 = images[1]

image2 = img.imread(path+'/'+image_name2)
ax = plt.subplot(2, 1, 2)
plt.imshow(image2)
plt.title(image_name2)
plt.axis("off")


print(images)


plt.savefig('/home/Thesis/PatientSampleImages.png')