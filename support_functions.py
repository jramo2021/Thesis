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


def visualize_augmentations(train_ds,rot_and_flip_aug,filename = "augmentations"):
    # Save sample images of training data
    plt.figure(figsize=(10, 10))
    image, label = next(iter(train_ds))

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
        plt.title('Possible Augment '+str(i+1))
        plt.axis("off")

    # Save image
    plt.savefig('/home/Thesis/'+filename+'.png')

def plot_history(history):

    plt.figure() 
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
    plt.savefig('/home/Thesis/Loss Plot.png')
    
    plt.figure()
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
    plt.savefig('/home/Thesis/Accuracy Plot.png')