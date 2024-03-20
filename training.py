"""
This module trains a logistic regression model using the provided dataset.
"""

import os
import json
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Load config.json and get path variables
with open('config.json', 'r', encoding='utf-8') as file:
    config = json.load(file)

dataset_csv_path = os.path.join(config['output_folder_path'], 'finaldata.csv')
model_path = os.path.join(config['output_model_path'], 'trainedmodel.pkl')


def train_model():
    """
    Trains a logistic regression model using the provided dataset.
    """
    # Read the dataset
    data = pd.read_csv(dataset_csv_path).iloc[:, 1:]

    # Split the dataset into features and target variable
    x = data.drop('exited', axis=1)
    y = data['exited']

    # Split the data into training and test sets
    x_train, _, y_train, _ = train_test_split(
        x, y, test_size=0.2, random_state=0)

    # Initialize the logistic regression model with the given hyperparameters
    model = LogisticRegression(
        C=1.0, class_weight=None, dual=False, fit_intercept=True,
        intercept_scaling=1, l1_ratio=None, max_iter=100,
        multi_class='auto', n_jobs=None, penalty='l2',
        random_state=0, solver='liblinear', tol=0.0001, verbose=0,
        warm_start=False
    )

    # Fit the logistic regression to the data
    model.fit(x_train, y_train)

    # Write the trained model to a file called trainedmodel.pkl
    with open(model_path, 'wb') as model_file:
        pickle.dump(model, model_file)


if __name__ == '__main__':
    train_model()
