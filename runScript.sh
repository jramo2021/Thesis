#!/bin/bash
python3 -c 'from data_management import *; get_data()'
cd /tmp/.keras/datasets
tfds new my_dataset
cd my_dataset
tfds build