# First XGBoost model for Pima Indians dataset
#from numpy import loadtxt
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
# load data
#dataset = loadtxt('train.csv', delimiter=",")
dataset = pd.read_csv("/content/train.csv")
# split data into X and y
# X_train = dataset[:,0:8]
# y_train = dataset[:,8]
x_tra = dataset['text']
y_train = dataset['label']

# Define the TF-IDF vectorizer for unigrams
tfidf_vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 1))

# Transform the text data to TF-IDF vectors
x_train = tfidf_vectorizer.fit_transform(x_tra)
# fit model no training data
model = XGBClassifier()

# Assuming you have found the best parameters and stored them in a dictionary called best_params
best_params = {'learning_rate': 0.1, 'n_estimators': 50, 'max_depth': 3, 'min_child_weight': 1}

# Initialize XGBClassifier with the best parameters
model = XGBClassifier(**best_params)

model.fit(x_train, y_train)

pickle.dump(model, open("pima_model.pkl", "wb"))
