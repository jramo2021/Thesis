import os, pathlib
import random
import shutil
import tensorflow as tf
import time
import glob

PatientArray = [
    "/tmp/.keras/datasets/BreaKHis_v1/histology_slides/breast/benign/SOB/adenosis/SOB_B_A_14-22549AB",
    "/tmp/.keras/datasets/BreaKHis_v1/histology_slides/breast/benign/SOB/adenosis/SOB_B_A_14-22549CD",
    "/tmp/.keras/datasets/BreaKHis_v1/histology_slides/breast/benign/SOB/adenosis/SOB_B_A_14-22549G",
    "/tmp/.keras/datasets/BreaKHis_v1/histology_slides/breast/benign/SOB/adenosis/SOB_B_A_14-29960CD",
    "/tmp/.keras/datasets/BreaKHis_v1/histology_slides/breast/benign/SOB/fibroadenoma/SOB_B_F_14-14134",
    "/tmp/.keras/datasets/BreaKHis_v1/histology_slides/breast/benign/SOB/fibroadenoma/SOB_B_F_14-21998EF",
    "/tmp/.keras/datasets/BreaKHis_v1/histology_slides/breast/benign/SOB/fibroadenoma/SOB_B_F_14-23222AB",
    "/tmp/.keras/datasets/BreaKHis_v1/histology_slides/breast/benign/SOB/fibroadenoma/SOB_B_F_14-9133",
    "/tmp/.keras/datasets/BreaKHis_v1/histology_slides/breast/benign/SOB/fibroadenoma/SOB_B_F_14-14134E",
    "/tmp/.keras/datasets/BreaKHis_v1/histology_slides/breast/benign/SOB/fibroadenoma/SOB_B_F_14-23060AB",
    "/tmp/.keras/datasets/BreaKHis_v1/histology_slides/breast/benign/SOB/fibroadenoma/SOB_B_F_14-25197",
    "/tmp/.keras/datasets/BreaKHis_v1/histology_slides/breast/benign/SOB/fibroadenoma/SOB_B_F_14-21998CD",
    "/tmp/.keras/datasets/BreaKHis_v1/histology_slides/breast/benign/SOB/fibroadenoma/SOB_B_F_14-23060CD",
    "/tmp/.keras/datasets/BreaKHis_v1/histology_slides/breast/benign/SOB/fibroadenoma/SOB_B_F_14-29960AB",
    "/tmp/.keras/datasets/BreaKHis_v1/histology_slides/breast/benign/SOB/phyllodes_tumor/SOB_B_PT_14-21998AB",
    "/tmp/.keras/datasets/BreaKHis_v1/histology_slides/breast/benign/SOB/phyllodes_tumor/SOB_B_PT_14-22704",
    "/tmp/.keras/datasets/BreaKHis_v1/histology_slides/breast/benign/SOB/phyllodes_tumor/SOB_B_PT_14-29315EF",
    "/tmp/.keras/datasets/BreaKHis_v1/histology_slides/breast/benign/SOB/tubular_adenoma/SOB_B_TA_14-13200",
    "/tmp/.keras/datasets/BreaKHis_v1/histology_slides/breast/benign/SOB/tubular_adenoma/SOB_B_TA_14-16184",
    "/tmp/.keras/datasets/BreaKHis_v1/histology_slides/breast/benign/SOB/tubular_adenoma/SOB_B_TA_14-19854C",
    "/tmp/.keras/datasets/BreaKHis_v1/histology_slides/breast/benign/SOB/tubular_adenoma/SOB_B_TA_14-3411F",
    "/tmp/.keras/datasets/BreaKHis_v1/histology_slides/breast/benign/SOB/tubular_adenoma/SOB_B_TA_14-15275",
    "/tmp/.keras/datasets/BreaKHis_v1/histology_slides/breast/benign/SOB/tubular_adenoma/SOB_B_TA_14-16184CD",
    "/tmp/.keras/datasets/BreaKHis_v1/histology_slides/breast/benign/SOB/tubular_adenoma/SOB_B_TA_14-21978AB",
    "/tmp/.keras/datasets/BreaKHis_v1/histology_slides/breast/malignant/SOB/ductal_carcinoma/SOB_M_DC_14-10926",
    "/tmp/.keras/datasets/BreaKHis_v1/histology_slides/breast/malignant/SOB/ductal_carcinoma/SOB_M_DC_14-15572",
    "/tmp/.keras/datasets/BreaKHis_v1/histology_slides/breast/malignant/SOB/ductal_carcinoma/SOB_M_DC_14-17901",
    "/tmp/.keras/datasets/BreaKHis_v1/histology_slides/breast/malignant/SOB/ductal_carcinoma/SOB_M_DC_14-4364",
    "/tmp/.keras/datasets/BreaKHis_v1/histology_slides/breast/malignant/SOB/ductal_carcinoma/SOB_M_DC_14-11031",
    "/tmp/.keras/datasets/BreaKHis_v1/histology_slides/breast/malignant/SOB/ductal_carcinoma/SOB_M_DC_14-15696",
    "/tmp/.keras/datasets/BreaKHis_v1/histology_slides/breast/malignant/SOB/ductal_carcinoma/SOB_M_DC_14-17915",
    "/tmp/.keras/datasets/BreaKHis_v1/histology_slides/breast/malignant/SOB/ductal_carcinoma/SOB_M_DC_14-4372",
    "/tmp/.keras/datasets/BreaKHis_v1/histology_slides/breast/malignant/SOB/ductal_carcinoma/SOB_M_DC_14-11520",
    "/tmp/.keras/datasets/BreaKHis_v1/histology_slides/breast/malignant/SOB/ductal_carcinoma/SOB_M_DC_14-15792",
    "/tmp/.keras/datasets/BreaKHis_v1/histology_slides/breast/malignant/SOB/ductal_carcinoma/SOB_M_DC_14-18650",
    "/tmp/.keras/datasets/BreaKHis_v1/histology_slides/breast/malignant/SOB/ductal_carcinoma/SOB_M_DC_14-5287",
    "/tmp/.keras/datasets/BreaKHis_v1/histology_slides/breast/malignant/SOB/ductal_carcinoma/SOB_M_DC_14-11951",
    "/tmp/.keras/datasets/BreaKHis_v1/histology_slides/breast/malignant/SOB/ductal_carcinoma/SOB_M_DC_14-16188",
    "/tmp/.keras/datasets/BreaKHis_v1/histology_slides/breast/malignant/SOB/ductal_carcinoma/SOB_M_DC_14-20629",
    "/tmp/.keras/datasets/BreaKHis_v1/histology_slides/breast/malignant/SOB/ductal_carcinoma/SOB_M_DC_14-5694",
    "/tmp/.keras/datasets/BreaKHis_v1/histology_slides/breast/malignant/SOB/ductal_carcinoma/SOB_M_DC_14-12312",
    "/tmp/.keras/datasets/BreaKHis_v1/histology_slides/breast/malignant/SOB/ductal_carcinoma/SOB_M_DC_14-16336",
    "/tmp/.keras/datasets/BreaKHis_v1/histology_slides/breast/malignant/SOB/ductal_carcinoma/SOB_M_DC_14-20636",
    "/tmp/.keras/datasets/BreaKHis_v1/histology_slides/breast/malignant/SOB/ductal_carcinoma/SOB_M_DC_14-5695",
    "/tmp/.keras/datasets/BreaKHis_v1/histology_slides/breast/malignant/SOB/ductal_carcinoma/SOB_M_DC_14-13412",
    "/tmp/.keras/datasets/BreaKHis_v1/histology_slides/breast/malignant/SOB/ductal_carcinoma/SOB_M_DC_14-16448",
    "/tmp/.keras/datasets/BreaKHis_v1/histology_slides/breast/malignant/SOB/ductal_carcinoma/SOB_M_DC_14-2523",
    "/tmp/.keras/datasets/BreaKHis_v1/histology_slides/breast/malignant/SOB/ductal_carcinoma/SOB_M_DC_14-6241",
    "/tmp/.keras/datasets/BreaKHis_v1/histology_slides/breast/malignant/SOB/ductal_carcinoma/SOB_M_DC_14-13993",
    "/tmp/.keras/datasets/BreaKHis_v1/histology_slides/breast/malignant/SOB/ductal_carcinoma/SOB_M_DC_14-16601",
    "/tmp/.keras/datasets/BreaKHis_v1/histology_slides/breast/malignant/SOB/ductal_carcinoma/SOB_M_DC_14-2773",
    "/tmp/.keras/datasets/BreaKHis_v1/histology_slides/breast/malignant/SOB/ductal_carcinoma/SOB_M_DC_14-8168",
    "/tmp/.keras/datasets/BreaKHis_v1/histology_slides/breast/malignant/SOB/ductal_carcinoma/SOB_M_DC_14-14015",
    "/tmp/.keras/datasets/BreaKHis_v1/histology_slides/breast/malignant/SOB/ductal_carcinoma/SOB_M_DC_14-16716",
    "/tmp/.keras/datasets/BreaKHis_v1/histology_slides/breast/malignant/SOB/ductal_carcinoma/SOB_M_DC_14-2980",
    "/tmp/.keras/datasets/BreaKHis_v1/histology_slides/breast/malignant/SOB/ductal_carcinoma/SOB_M_DC_14-9461",
    "/tmp/.keras/datasets/BreaKHis_v1/histology_slides/breast/malignant/SOB/ductal_carcinoma/SOB_M_DC_14-14926",
    "/tmp/.keras/datasets/BreaKHis_v1/histology_slides/breast/malignant/SOB/ductal_carcinoma/SOB_M_DC_14-16875",
    "/tmp/.keras/datasets/BreaKHis_v1/histology_slides/breast/malignant/SOB/ductal_carcinoma/SOB_M_DC_14-2985",
    "/tmp/.keras/datasets/BreaKHis_v1/histology_slides/breast/malignant/SOB/ductal_carcinoma/SOB_M_DC_14-14946",
    "/tmp/.keras/datasets/BreaKHis_v1/histology_slides/breast/malignant/SOB/ductal_carcinoma/SOB_M_DC_14-17614",
    "/tmp/.keras/datasets/BreaKHis_v1/histology_slides/breast/malignant/SOB/ductal_carcinoma/SOB_M_DC_14-3909",
    "/tmp/.keras/datasets/BreaKHis_v1/histology_slides/breast/malignant/SOB/lobular_carcinoma/SOB_M_LC_14-12204",
    "/tmp/.keras/datasets/BreaKHis_v1/histology_slides/breast/malignant/SOB/lobular_carcinoma/SOB_M_LC_14-15570",
    "/tmp/.keras/datasets/BreaKHis_v1/histology_slides/breast/malignant/SOB/lobular_carcinoma/SOB_M_LC_14-16196",
    "/tmp/.keras/datasets/BreaKHis_v1/histology_slides/breast/malignant/SOB/lobular_carcinoma/SOB_M_LC_14-13412",
    "/tmp/.keras/datasets/BreaKHis_v1/histology_slides/breast/malignant/SOB/lobular_carcinoma/SOB_M_LC_14-15570C",
    "/tmp/.keras/datasets/BreaKHis_v1/histology_slides/breast/malignant/SOB/mucinous_carcinoma/SOB_M_MC_14-10147",
    "/tmp/.keras/datasets/BreaKHis_v1/histology_slides/breast/malignant/SOB/mucinous_carcinoma/SOB_M_MC_14-13418DE",
    "/tmp/.keras/datasets/BreaKHis_v1/histology_slides/breast/malignant/SOB/mucinous_carcinoma/SOB_M_MC_14-18842D",
    "/tmp/.keras/datasets/BreaKHis_v1/histology_slides/breast/malignant/SOB/mucinous_carcinoma/SOB_M_MC_14-12773",
    "/tmp/.keras/datasets/BreaKHis_v1/histology_slides/breast/malignant/SOB/mucinous_carcinoma/SOB_M_MC_14-16456",
    "/tmp/.keras/datasets/BreaKHis_v1/histology_slides/breast/malignant/SOB/mucinous_carcinoma/SOB_M_MC_14-19979",
    "/tmp/.keras/datasets/BreaKHis_v1/histology_slides/breast/malignant/SOB/mucinous_carcinoma/SOB_M_MC_14-13413",
    "/tmp/.keras/datasets/BreaKHis_v1/histology_slides/breast/malignant/SOB/mucinous_carcinoma/SOB_M_MC_14-18842",
    "/tmp/.keras/datasets/BreaKHis_v1/histology_slides/breast/malignant/SOB/mucinous_carcinoma/SOB_M_MC_14-19979C",
    "/tmp/.keras/datasets/BreaKHis_v1/histology_slides/breast/malignant/SOB/papillary_carcinoma/SOB_M_PC_14-12465",
    "/tmp/.keras/datasets/BreaKHis_v1/histology_slides/breast/malignant/SOB/papillary_carcinoma/SOB_M_PC_14-15704",
    "/tmp/.keras/datasets/BreaKHis_v1/histology_slides/breast/malignant/SOB/papillary_carcinoma/SOB_M_PC_14-9146",
    "/tmp/.keras/datasets/BreaKHis_v1/histology_slides/breast/malignant/SOB/papillary_carcinoma/SOB_M_PC_14-15687B",
    "/tmp/.keras/datasets/BreaKHis_v1/histology_slides/breast/malignant/SOB/papillary_carcinoma/SOB_M_PC_14-19440",
    "/tmp/.keras/datasets/BreaKHis_v1/histology_slides/breast/malignant/SOB/papillary_carcinoma/SOB_M_PC_15-190EF",
]   

subclassPathArray = [
    #"/tmp/.keras/datasets/BreaKHis_v1/histology_slides/breast/benign/SOB/adenosis/",
    "/tmp/.keras/datasets/BreaKHis_v1/histology_slides/breast/benign/SOB/fibroadenoma/",
    "/tmp/.keras/datasets/BreaKHis_v1/histology_slides/breast/benign/SOB/phyllodes_tumor/",
    "/tmp/.keras/datasets/BreaKHis_v1/histology_slides/breast/benign/SOB/tubular_adenoma/",
    "/tmp/.keras/datasets/BreaKHis_v1/histology_slides/breast/malignant/SOB/ductal_carcinoma/",
    "/tmp/.keras/datasets/BreaKHis_v1/histology_slides/breast/malignant/SOB/lobular_carcinoma/",
    "/tmp/.keras/datasets/BreaKHis_v1/histology_slides/breast/malignant/SOB/mucinous_carcinoma/",
    "/tmp/.keras/datasets/BreaKHis_v1/histology_slides/breast/malignant/SOB/papillary_carcinoma/"
]

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

# destPath = "/tmp/.keras/datasets/BalancedDataset"
# if not os.path.isdir(destPath):
#     os.mkdir(destPath)
#     os.mkdir(destPath + '/benign')
#     os.mkdir(destPath + '/malignant')
# for i in subclassPathArray:
#     subclassPath = pathlib.Path(i)
#     subclassImageList = list(subclassPath.glob('**/*.png'))
#     for j in range(0, 1000):
#         randNum = random.randrange(0, len(subclassImageList))
#         if 'benign' in i:
#             #currImgPath = subclassImageList[j].as_posix
#             shutil.copy(str(subclassImageList[randNum]), destPath + '/benign/BenignSample' + str(j) + '.png')
#         else:
#             #currImgPath = subclassImageList[j].as_posix()
#             shutil.copy(str(subclassImageList[randNum]), destPath + '/malignant/MalignantSample' + str(j) + '.png')

# destPath = "/tmp/.keras/BalancedDataset"
# if not os.path.isdir(destPath):
#     os.mkdir(destPath)
#     os.mkdir(destPath + '/benign')
#     os.mkdir(destPath + '/malignant')
# for i in subclassPathArray:
#     subclassPath = pathlib.Path(i)
#     subclassImageList = list(subclassPath.glob('**/*.png'))
#     random.shuffle(subclassImageList)
#     count = 1
#     for k in subclassImageList:
#         if 'adenosis' in i:
#             shutil.copy(str(k), destPath + '/benign/AdenosisSample' + str(count) + '.png')
#         elif 'fibro' in i:
#             shutil.copy(str(k), destPath + '/benign/FibroadenomaSample' + str(count) + '.png')
#         elif 'phyllodes' in i:
#             shutil.copy(str(k), destPath + '/benign/PhyllodesSample' + str(count) + '.png')
#         elif 'tubular' in i:
#             shutil.copy(str(k), destPath + '/benign/TubularSample' + str(count) + '.png')
#         elif 'ductal' in i:
#             shutil.copy(str(k), destPath + '/malignant/DuctalSample' + str(count) + '.png')
#         elif 'lobular' in i:
#             shutil.copy(str(k), destPath + '/malignant/LobularSample' + str(count) + '.png')
#         elif 'mucinous' in i:
#             shutil.copy(str(k), destPath + '/malignant/MucinousSample' + str(count) + '.png')
#         else:
#             shutil.copy(str(k), destPath + '/malignant/PapillarySample' + str(count) + '.png')
#         count = count + 1
#         if count == 1001:
#             break

#     for j in range(0, 1000 - len(subclassImageList)):
#         randNum = random.randrange(0, len(subclassImageList))
#         if count == 1001:
#             break
#         if 'adenosis' in i:
#             shutil.copy(str(subclassImageList[randNum]), destPath + '/benign/AdenosisSample' + str(len(subclassImageList)+j+1) + '.png')
#         elif 'fibro' in i:
#             shutil.copy(str(subclassImageList[randNum]), destPath + '/benign/FibroadenomaSample' + str(len(subclassImageList)+j+1) + '.png')
#         elif 'phyllodes' in i:
#             shutil.copy(str(subclassImageList[randNum]), destPath + '/benign/PhyllodesSample' + str(len(subclassImageList)+j+1) + '.png')
#         elif 'tubular' in i:
#             shutil.copy(str(subclassImageList[randNum]), destPath + '/benign/TubularSample' + str(len(subclassImageList)+j+1) + '.png')
#         elif 'ductal' in i:
#             shutil.copy(str(subclassImageList[randNum]), destPath + '/malignant/DuctalSample' + str(len(subclassImageList)+j+1) + '.png')
#         elif 'lobular' in i:
#             shutil.copy(str(subclassImageList[randNum]), destPath + '/malignant/LobularSample' + str(len(subclassImageList)+j+1) + '.png')
#         elif 'mucinous' in i:
#             shutil.copy(str(subclassImageList[randNum]), destPath + '/malignant/MucinousSample' + str(len(subclassImageList)+j+1) + '.png')
#         else:
#             shutil.copy(str(subclassImageList[randNum]), destPath + '/malignant/PapillarySample' + str(len(subclassImageList)+j+1) + '.png')


# shutil.rmtree("/tmp/.keras/datasets/")

destPath = "/tmp/.keras/AugmentedDataset"
if not os.path.isdir(destPath):
    os.mkdir(destPath)
    os.mkdir(destPath + '/benign')
    os.mkdir(destPath + '/malignant')

for i in subclassPathArray:
    subclassPath = pathlib.Path(i)
    subclassImageList = list(subclassPath.glob('**/*.png'))
    random.shuffle(subclassImageList)

    for j in range(0, 444):
        randNum = random.sample(range(len(subclassImageList)), k=len(subclassImageList))
        # if 'adenosis' in i:
        #     shutil.copy(str(subclassImageList[randNum]), destPath + '/benign/AdenosisSample' + str(len(subclassImageList)+j+1) + '.png')
        if 'fibro' in i:
            shutil.copy(str(subclassImageList[randNum[j]]), destPath + '/benign/FibroadenomaSample' + str(j+1) + '.png')
        elif 'phyllodes' in i:
            shutil.copy(str(subclassImageList[randNum[j]]), destPath + '/benign/PhyllodesSample' + str(j+1) + '.png')
        elif 'tubular' in i:
            shutil.copy(str(subclassImageList[randNum[j]]), destPath + '/benign/TubularSample' + str(j+1) + '.png')
        elif 'ductal' in i:
            shutil.copy(str(subclassImageList[randNum[j]]), destPath + '/malignant/DuctalSample' + str(j+1) + '.png')
        elif 'lobular' in i:
            shutil.copy(str(subclassImageList[randNum[j]]), destPath + '/malignant/LobularSample' + str(j+1) + '.png')
        elif 'mucinous' in i:
            shutil.copy(str(subclassImageList[randNum[j]]), destPath + '/malignant/MucinousSample' + str(j+1) + '.png')
        elif 'papillary' in i:
            shutil.copy(str(subclassImageList[randNum[j]]), destPath + '/malignant/PapillarySample' + str(j+1) + '.png')
    
    AdenosisPath = pathlib.Path("/tmp/.keras/datasets/BreaKHis_v1/histology_slides/breast/benign/SOB/adenosis/")
    AdenosisImageList = list(AdenosisPath.glob('**/*.png'))
    for i in range(0, 444):
        shutil.copy(str(AdenosisImageList[i]), destPath + '/benign/AdenosisSample' + str(i+1) + '.png')

#do not remove this line or server will crash
shutil.rmtree("/tmp/.keras/datasets/")

# makes aug data. if 2 datasets made from both directories and 
# data in augmented data is augmented then makes a dataset with 
# 3451 (# of samples in biggest class DuctalC) samples in each class

# destPath = "/tmp/.keras/AugmentedData"
#         if not os.path.isdir(destPath):
#             os.mkdir(destPath)
#             os.mkdir(destPath + '/benign')
#             os.mkdir(destPath + '/malignant')

#         for i in subclassPathArray:
#             subclassPath = pathlib.Path(i)
#             subclassImageList = list(subclassPath.glob('**/*.png'))
#             random.shuffle(subclassImageList)

#             for j in range(0, 3451 - len(subclassImageList)):
#                 randNum = random.randrange(0, len(subclassImageList))
#                 if 'adenosis' in i:
#                     shutil.copy(str(subclassImageList[randNum]), destPath + '/benign/AdenosisSample' + str(len(subclassImageList)+j+1) + '.png')
#                 elif 'fibro' in i:
#                     shutil.copy(str(subclassImageList[randNum]), destPath + '/benign/FibroadenomaSample' + str(len(subclassImageList)+j+1) + '.png')
#                 elif 'phyllodes' in i:
#                     shutil.copy(str(subclassImageList[randNum]), destPath + '/benign/PhyllodesSample' + str(len(subclassImageList)+j+1) + '.png')
#                 elif 'tubular' in i:
#                     shutil.copy(str(subclassImageList[randNum]), destPath + '/benign/TubularSample' + str(len(subclassImageList)+j+1) + '.png')
#                 elif 'ductal' in i:
#                     shutil.copy(str(subclassImageList[randNum]), destPath + '/malignant/DuctalSample' + str(len(subclassImageList)+j+1) + '.png')
#                 elif 'lobular' in i:
#                     shutil.copy(str(subclassImageList[randNum]), destPath + '/malignant/LobularSample' + str(len(subclassImageList)+j+1) + '.png')
#                 elif 'mucinous' in i:
#                     shutil.copy(str(subclassImageList[randNum]), destPath + '/malignant/MucinousSample' + str(len(subclassImageList)+j+1) + '.png')
#                 else:
#                     shutil.copy(str(subclassImageList[randNum]), destPath + '/malignant/PapillarySample' + str(len(subclassImageList)+j+1) + '.png')