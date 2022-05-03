import unittest
import numpy as np
from main import *

class TestList(unittest.TestCase):
    
    '''Testing that train_ds is consistently taken from preprocessing'''
    def test_preprocess_data_0(self):
        '''Collect and Preprocess Data'''
        # Download Data
        data_dir = get_data()
        
        # Preprocess Data (No data augmentation)
        train_ds, val_ds, test_ds = preprocess_data(data_dir,aug_split = 0)
        ds1 = iter(train_ds)
        train_ds, val_ds, test_ds = preprocess_data(data_dir,aug_split = 0)
        ds2 = iter(train_ds)
        
        for i in range(len(train_ds)):
            images, labels1 = next(ds1)
            images, labels2 = next(ds2)
            
            labels1 = tf.constant(labels1).numpy()
            labels2 = tf.constant(labels2).numpy()
            # print(labels1) 
            # print(labels2)
            
            self.assertTrue(np.array_equal(labels1,labels2))
        
    # '''Testing that val_ds is consistently taken from preprocessing'''
    # def test_preprocess_data_1(self):
    #     '''Collect and Preprocess Data'''
    #     # Download Data
    #     data_dir = get_data()
        
    #     # Preprocess Data (No data augmentation)
    #     train_ds, val_ds, test_ds = preprocess_data(data_dir,aug_split = 0)
    #     ds1 = iter(val_ds)
    #     train_ds, val_ds, test_ds = preprocess_data(data_dir,aug_split = 0)
    #     ds2 = iter(val_ds)
        
    #     for i in range(len(val_ds)):
    #         images, labels1 = next(ds1)
    #         images, labels2 = next(ds2)
            
    #         labels1 = tf.constant(labels1).numpy()
    #         labels2 = tf.constant(labels2).numpy()
    #         # print(labels1) 
    #         # print(labels2)
            
    #         self.assertTrue(np.array_equal(labels1,labels2))

    # '''Testing that test_ds is consistently taken from preprocessing'''
    # def test_preprocess_data_2(self):
    #     '''Collect and Preprocess Data'''
    #     # Download Data
    #     data_dir = get_data()
        
    #     # Preprocess Data (No data augmentation)
    #     train_ds, val_ds, test_ds = preprocess_data(data_dir,aug_split = 0)
    #     ds1 = iter(test_ds)
    #     train_ds, val_ds, test_ds = preprocess_data(data_dir,aug_split = 0)
    #     ds2 = iter(test_ds)
        
    #     print("First and Last 5 values of Test Labels")
    #     for i in range(len(test_ds)):
    #         images, labels1 = next(ds1)
    #         images, labels2 = next(ds2)
            
    #         labels1 = tf.constant(labels1).numpy()
    #         labels2 = tf.constant(labels2).numpy()
    #         # print(labels1) 
    #         # print(labels2)
            
    #         self.assertTrue(np.array_equal(labels1,labels2))

    # ''' Since the training set is the last dataset to be partitioned,
    #     the last batch may not be full. The full dataset size is 7909 samples,
    #     so dividing samples into batches of size of 32 will lead to a nonfull batch.
    #     This is expected. This double checks that the last batch is the expected size.'''
    # def test_preprocess_data_3(self):
    #     '''Collect and Preprocess Data'''
    #     # Download Data
    #     data_dir = get_data()
        
    #     # Preprocess Data (No data augmentation)
    #     train_ds, val_ds, test_ds = preprocess_data(data_dir,aug_split = 0)
    #     train_ds = iter(train_ds)

    #     # Get to the labels of the last batch
    #     for image, last in train_ds:
    #         continue
        
    #     # Convert Tensor object to number array
    #     last = tf.constant(last).numpy()
        
    #     # Check that last batch is proper size
    #     self.assertEqual(len(last),7909%32)
        
    # '''Checks that the ratio is consistent with different augmentation splits'''
    # def test_preprocess_data_4(self):
    #     '''Collect and Preprocess Data'''
    #     # Download Data
    #     data_dir = get_data()
        
    #     for aug_split in np.arange(0,1.1,.1):
    #         # Preprocess Data (No data augmentation)
    #         train_ds, val_ds, test_ds = preprocess_data(data_dir,aug_split = aug_split)
            
    #         # Calculate size (ropund to nearest 10th because it won't be exact)
    #         total_size = len(train_ds) + len(val_ds) + len(test_ds)
    #         train_len = round(len(train_ds)/(total_size), 1)
    #         val_len = round(len(val_ds)/(total_size), 1)
    #         test_len = round(len(test_ds)/(total_size), 1)
            
    #         # Tests that train, val, test split is .8:.1:.1
    #         self.assertAlmostEqual(train_len,.8)
    #         self.assertAlmostEqual(val_len,.1)
    #         self.assertAlmostEqual(test_len,.1)

    
if __name__ == '__main__':
   unittest.main()