# First XGBoost model for Pima Indians dataset
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from numpy import loadtxt
import pickle
# load data
#dataset = loadtxt('/content/python-xgboost-basic-master/pima-indians-diabetes.data', delimiter=",")
dataset= pd.read_excel("/content/python-xgboost-basic-master/FN_dataset(12000).xlsx")
seed = 7
test_size = 0.3

a_train,a_test  = train_test_split(dataset, test_size=test_size, random_state=seed)
# np.savetxt("train.csv",a_train,delimiter=",")
# np.savetxt("test.csv",a_test,delimiter=",")
a_train.to_csv('train.csv', index=False)
a_test.to_csv('test.csv', index=False)