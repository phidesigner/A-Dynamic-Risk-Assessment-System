"""
This module contains code for scoring a trained model.
"""

import os
import pickle
import json
import pandas as pd
from sklearn import metrics


# Load config.json and get path variables
with open('config.json', 'r', encoding='utf-8') as f:
    config = json.load(f)

test_data_path = os.path.join(config['test_data_path'], 'testdata.csv')
model_path = os.path.join(config['output_model_path'], 'trainedmodel.pkl')
score_file_path = os.path.join(config['output_model_path'], 'latestscore.txt')

# Function for model scoring


def score_model():
    '''
    This function takes a trained model, load test data, and calculate
    an F1 score for the model relative to the test data
    '''
    # Read in test data
    test_data = pd.read_csv(test_data_path).iloc[:, 1:]

    x_test = test_data.drop('exited', axis=1)
    y_test = test_data['exited']

    # Load the trained ML model
    with open(model_path, 'rb') as model_file:
        model = pickle.load(model_file)

    # Predict on the test data
    y_pred = model.predict(x_test)

    # Calculate F1 score
    f1 = metrics.f1_score(y_test, y_pred)

    # Write the F1 score to the latestscore.txt file
    with open(score_file_path, 'w', encoding='utf-8') as score_file:
        score_file.write(f"F1 Score: {f1}")

    return f1


if __name__ == '__main__':
    score_model()
