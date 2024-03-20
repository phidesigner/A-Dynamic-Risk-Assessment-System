
"""
This module contains a function to score a model
and generate a confusion matrix plot.
"""

import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from diagnostics import model_predictions

# Load config.json and get path variables
with open('config.json', 'r', encoding='utf-8') as f:
    config = json.load(f)

dataset_csv_path = os.path.join(config['output_folder_path'])
output_model_path = os.path.join(config['output_model_path'])
test_data_path = os.path.join(config['test_data_path'], 'testdata.csv')

# Function for reporting


def score_model():
    """
    Function to score the model and generate a confusion matrix plot.

    This function loads the model predictions, the actual labels from the test
    data, calculates a confusion matrix using the test data and the
    predictions, and plots the confusion matrix using a heatmap.
    The confusion matrix plot is then saved to a file.
    """
    # Get model predictions
    predictions = model_predictions(test_data_path)

    # Load the actual labels
    test_data = pd.read_csv(test_data_path)
    actual = test_data['exited']

    # Calculate a confusion matrix using the test data
    # and the deployed model predictions
    cm = metrics.confusion_matrix(actual, predictions)

    # Plotting the confusion matrix
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt="d")
    plt.title('Confusion Matrix')
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')

    # Save the confusion matrix plot
    confusion_matrix_plot_path = os.path.join(
        output_model_path, 'confusionmatrix.png')
    plt.savefig(confusion_matrix_plot_path)
    print(f"Confusion matrix plot saved to {confusion_matrix_plot_path}")


if __name__ == '__main__':
    score_model()
