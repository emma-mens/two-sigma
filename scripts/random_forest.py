#!/data/sls/u/meng/emazuh/anaconda3/bin/python3

#SBATCH --output=rf_output.log
#SBATCH -n 50 # cores
#SBATCH -p 630
#SBATCH --job-name=rf
#SBATCH --mem=12000

import pickle
import numpy as np
import pandas as pd
import ast

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor


data_file = '/data/sls/temp/emazuh/867/data.p'
augmented_data_file = '/data/sls/temp/emazuh/867/augmented_dataframe'
print('done loading data')

with open(data_file, 'rb') as f:
    train, test = pickle.load(f)

market_train, news_train = train
corpus = news_train.headline.values
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(corpus)
print('done vectorizing data')

augmented = pd.read_csv(augmented_data_file)
market_train = augmented[augmented.ndays == '1 days']

market_train.loc[:,'indices'] = market_train.indices.apply(lambda x: ast.literal_eval(x))
market_train.loc[:,'n_articles'] = market_train.indices.apply(lambda x: len(x))

def flatten(array):
    return [j for sub in array for j in sub]


data = market_train[market_train.n_articles == 1]

features = X[np.array(flatten(data.indices))]
labels = data.returnsOpenNextMktres10

rf = RandomForestRegressor(n_estimators = 200, max_depth=(features.shape[1])**(.5), verbose=100, n_jobs=-1)
train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.25, random_state = 42)
print('Training Features Shape:', train_features.shape)
print('Training Labels Shape:', train_labels.shape)
print('Testing Features Shape:', test_features.shape)
print('Testing Labels Shape:', test_labels.shape)

# The baseline predictions are the historical averages
baseline_preds = test_labels.mean()
# Baseline errors, and display average baseline error
baseline_errors = abs(baseline_preds - test_labels)
print('Average baseline error: ', round(np.mean(baseline_errors), 3))

print('started fitting')
rf.fit(train_features, train_labels)
print('done fitting')

with open('random_forest.p', 'wb') as fp:
    pickle.dump(rf, fp)

# Get numerical feature importances
importances = list(rf.feature_importances_)
# List of tuples with variable and importance
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
# Sort the feature importances by most important first
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
# Print out the feature and importances 
[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances];