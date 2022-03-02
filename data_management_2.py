import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

data = pd.read_csv('filepath')
display(data)

[new cell]

X_train, y_train, X_test, y_test = train_test_split(data['X'], data['Y'], random_seed=98)