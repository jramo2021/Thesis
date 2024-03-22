from AugmentData import rotate_and_flip_images
from AugmentData import split_image
from matplotlib import pyplot as plt
import cv2
import os

# img1, img2, img3 = rotate_and_flip_images('/tmp/.keras/datasets/100X/Train/malignant/SOB_M_DC_14-10926/SOB_M_DC-14-10926-100-001.png')

# original = cv2.imread('/tmp/.keras/datasets/100X/Train/malignant/SOB_M_DC_14-10926/SOB_M_DC-14-10926-100-001.png')
# cv2.imwrite('/home/Thesis/original.png', original)
# cv2.imwrite('img1.png',img1)
# cv2.imwrite('img2.png',img2)
# cv2.imwrite('img3.png',img3)

# patches = split_image('/tmp/.keras/datasets/100X/Train/malignant/SOB_M_DC_14-10926/SOB_M_DC-14-10926-100-001.png')
# count = 0
# for patch in patches:
#     cv2.imwrite('patch' + str(count) +'.png',patch)
#     count += 1

mag = 'All'

dir_b = '/tmp/.keras/datasets/' + mag + '/Train/' + 'benign'
dir_m = '/tmp/.keras/datasets/' + mag + '/Train/' + 'malignant'

if mag == 'All':
    for patient in os.listdir(dir_b):
        for mag in os.listdir(os.path.join(dir_b, patient)):
            for img in os.listdir(os.path.join(dir_b, patient, mag)):
                if 'aug' not in img and 'sub' not in img:
                    path2img = os.path.join(dir_b, patient, mag, img)
                    aug1, aug2, aug3 = rotate_and_flip_images(path2img)
                    split = path2img.split('.')
                    path2img = '.'.join(split[0:2])
                    cv2.imwrite(path2img + 'aug1.png', aug1)
                    cv2.imwrite(path2img + 'aug2.png', aug2)
                    cv2.imwrite(path2img + 'aug3.png', aug3)

                    image_path = os.path.join(dir_b, patient, mag, img)
                    subimages = split_image(image_path)
                    count = 0
                    for img in subimages:
                        cv2.imwrite(path2img + 'subimg' + str(count) + '.png', img)
                        count += 1

    for patient in os.listdir(dir_m):
        for mag in os.listdir(os.path.join(dir_m, patient)):
            for img in os.listdir(os.path.join(dir_m, patient, mag)):
                if 'aug' not in img and 'sub' not in img:
                    path2img = os.path.join(dir_m, patient, mag, img)
                    aug1, aug2, aug3 = rotate_and_flip_images(path2img)
                    split = path2img.split('.')
                    path2img = '.'.join(split[0:2])
                    cv2.imwrite(path2img + 'aug1.png', aug1)
                    cv2.imwrite(path2img + 'aug2.png', aug2)
                    cv2.imwrite(path2img + 'aug3.png', aug3)

                    image_path = os.path.join(dir_m, patient, mag, img)
                    subimages = split_image(image_path)
                    count = 0
                    for img in subimages:
                        cv2.imwrite(path2img + 'subimg' + str(count) + '.png', img)
                        count += 1
else:

    # augment data
    for patient in os.listdir(dir_b):
        for img in os.listdir(os.path.join(dir_b, patient)):
            if 'aug' not in img and 'sub' not in img:
                path2img = os.path.join(dir_b, patient, img)
                aug1, aug2, aug3 = rotate_and_flip_images(path2img)
                split = path2img.split('.')
                path2img = '.'.join(split[0:2])
                cv2.imwrite(path2img + 'aug1.png', aug1)
                cv2.imwrite(path2img + 'aug2.png', aug2)
                cv2.imwrite(path2img + 'aug3.png', aug3)

                image_path = os.path.join(dir_b, patient, img)
                subimages = split_image(image_path)
                count = 0
                for img in subimages:
                    cv2.imwrite(path2img + 'subimg' + str(count) + '.png', img)
                    count += 1

    for patient in os.listdir(dir_m):
        for img in os.listdir(os.path.join(dir_m, patient)):
            if 'aug'not in img and 'sub' not in img:
                path2img = os.path.join(dir_m, patient, img)
                aug1, aug2, aug3 = rotate_and_flip_images(path2img)
                split = path2img.split('.')
                path2img = '.'.join(split[0:2])
                cv2.imwrite(path2img + 'aug1.png', aug1)
                cv2.imwrite(path2img + 'aug2.png', aug2)
                cv2.imwrite(path2img + 'aug3.png', aug3)

                image_path = os.path.join(dir_m, patient, img)
                subimages = split_image(image_path)
                count = 0
                for img in subimages:
                    cv2.imwrite(path2img + 'subimg' + str(count) + '.png', img)
                    count += 1