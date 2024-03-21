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
model_path = os.path.join(config['prod_deployment_path'], 'trainedmodel.pkl')
score_file_path = os.path.join(
    config['prod_deployment_path'], 'latestscore.txt')

# Function for model scoring


def score_model(predictions=None, new_data_path=None):
    """
    Calculate the F1 score for a set of predictions against a given dataset,
    or against a default test dataset if no arguments are provided.

    Args:
    - predictions: Optional; A list/array of model predictions.
    - new_data_path: Optional; Path to a new data CSV file for scoring.

    Returns:
    - The F1 score as a float.
    """
    if predictions is None or new_data_path is None:
        # Default behavior: use the predefined test dataset and model
        test_data = pd.read_csv(test_data_path).iloc[:, 1:]
        x_test = test_data.drop('exited', axis=1)
        y_test = test_data['exited']

        # Load the trained ML model if predictions are not provided
        with open(model_path, 'rb') as model_file:
            model = pickle.load(model_file)
        predictions = model.predict(x_test)

        # Calculate F1 score
        f1 = metrics.f1_score(y_test, predictions)

        # Overwrite the score file only when using the default test dataset
        with open(score_file_path, 'w', encoding='utf-8') as score_file:
            score_file.write(f"F1 Score: {f1}")

    else:
        # Use provided predictions and new data for scoring
        new_data = pd.read_csv(new_data_path).iloc[:, 1:]
        # Adjust this to match the label column in your new dataset
        y_test = new_data['exited']

        # Calculate F1 score
        f1 = metrics.f1_score(y_test, predictions)

    return f1


if __name__ == '__main__':
    score_model()
