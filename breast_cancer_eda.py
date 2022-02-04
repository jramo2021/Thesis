import pandas as pd, numpy as np
import os, shutil
from glob import glob
from tqdm import tqdm
# import cv2
import matplotlib.pyplot as plt
# import seaborn as sns
# from colorama import Fore, Back, Style
# sns.set(style='dark')

# Example from https://www.kaggle.com/awsaf49/breast-cancer-eda
dataset_dir = '/kaggle/input/cbis-ddsm-breast-cancer-image-dataset'

df = pd.read_csv(f'{dataset_dir}/csv/dicom_info.csv')
df['image_path'] = df.image_path.apply(lambda x: x.replace('CBIS-DDSM', dataset_dir))
df.head()


def show_img(path):
    img = cv2.imread(path,0)
    plt.figure(figsize=(10,10))
    plt.imshow(img,cmap='bone')

show_img(df.image_path.iloc[10])


# %%time
import imagesize
data = df['image_path'].map(lambda path: imagesize.get(path))
width, height = list(zip(*data))
df['width'] = width
df['height'] = height
# df.head()


plt.figure(figsize=(12,8))
sns.kdeplot(df['width'], shade=True, color='limegreen')
sns.kdeplot(df['height'], shade=True, color='gold')
plt.legend(['width','height'])