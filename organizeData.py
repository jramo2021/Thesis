import os, pathlib
import random
import shutil
import tensorflow as tf
import time

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

if not os.path.isdir("/tmp/.keras/datasets/TrainTestValData"): 
    os.mkdir("/tmp/.keras/datasets/TrainTestValData")
    os.mkdir("/tmp/.keras/datasets/TrainTestValData/Test")
    os.mkdir("/tmp/.keras/datasets/TrainTestValData/Train")
    os.mkdir("/tmp/.keras/datasets/TrainTestValData/Val")
    os.mkdir("/tmp/.keras/datasets/TrainTestValData/Test/benign")
    os.mkdir("/tmp/.keras/datasets/TrainTestValData/Test/malignant")
    os.mkdir("/tmp/.keras/datasets/TrainTestValData/Train/benign")
    os.mkdir("/tmp/.keras/datasets/TrainTestValData/Train/malignant")
    os.mkdir("/tmp/.keras/datasets/TrainTestValData/Val/benign")
    os.mkdir("/tmp/.keras/datasets/TrainTestValData/Val/malignant")
else:
    shutil.rmtree("/tmp/.keras/datasets/TrainTestValData")
    os.mkdir("/tmp/.keras/datasets/TrainTestValData")
    os.mkdir("/tmp/.keras/datasets/TrainTestValData/Test")
    os.mkdir("/tmp/.keras/datasets/TrainTestValData/Train")
    os.mkdir("/tmp/.keras/datasets/TrainTestValData/Val")
    os.mkdir("/tmp/.keras/datasets/TrainTestValData/Test/benign")
    os.mkdir("/tmp/.keras/datasets/TrainTestValData/Test/malignant")
    os.mkdir("/tmp/.keras/datasets/TrainTestValData/Train/benign")
    os.mkdir("/tmp/.keras/datasets/TrainTestValData/Train/malignant")
    os.mkdir("/tmp/.keras/datasets/TrainTestValData/Val/benign")
    os.mkdir("/tmp/.keras/datasets/TrainTestValData/Val/malignant")

order = random.sample(range(82), k=82)
#print(order)
#print(len(order))

count = 0
for i in order:
    currPatient = PatientArray[i]
    allfiles = os.walk(currPatient)
    if count < 66:
        if "benign" in currPatient:
            for (root, dirs, files) in allfiles:
                for name in files:
                    src = os.path.join(root, name)
                    dst = os.path.join("/tmp/.keras/datasets/TrainTestValData/Train/benign", name)
                    shutil.copyfile(src, dst)
        else:
            for (root, dirs, files) in allfiles:
                for name in files:
                    src = os.path.join(root, name)
                    dst = os.path.join("/tmp/.keras/datasets/TrainTestValData/Train/malignant", name)
                    shutil.copyfile(src, dst)
    elif count >= 66 and count < 74:
        if "benign" in currPatient:
            for (root, dirs, files) in allfiles:
                for name in files:
                    src = os.path.join(root, name)
                    dst = os.path.join("/tmp/.keras/datasets/TrainTestValData/Test/benign", name)
                    shutil.copyfile(src, dst)
        else:
            for (root, dirs, files) in allfiles:
                for name in files:
                    src = os.path.join(root, name)
                    dst = os.path.join("/tmp/.keras/datasets/TrainTestValData/Test/malignant", name)
                    shutil.copyfile(src, dst)
    elif count >= 74:
        if "benign" in currPatient:
            for (root, dirs, files) in allfiles:
                for name in files:
                    src = os.path.join(root, name)
                    dst = os.path.join("/tmp/.keras/datasets/TrainTestValData/Val/benign", name)
                    shutil.copyfile(src, dst)
        else:
            for (root, dirs, files) in allfiles:
                for name in files:
                    src = os.path.join(root, name)
                    dst = os.path.join("/tmp/.keras/datasets/TrainTestValData/Val/malignant", name)
                    shutil.copyfile(src, dst)
    count += 1
