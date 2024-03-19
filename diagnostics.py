
"""
This module contains functions for diagnostics and monitoring
of a machine learning model.
"""

import os
import json
import pickle
import subprocess
import time
import pandas as pd

# Load config.json and get environment variables
with open('config.json', 'r', encoding='utf-8') as config_file:
    config = json.load(config_file)

dataset_csv_path = os.path.join(config['output_folder_path'])
test_data_path = os.path.join(config['test_data_path'])
prod_deployment_path = os.path.join(config['prod_deployment_path'])


# Function to get model predictions


def model_predictions():
    """
    Get model predictions for the given inference data.

    Args:
    - infer_data_path: Path to the inference data CSV file.

    Returns:
    - A list of model predictions.
    """
    infer_data_path = os.path.join(test_data_path, 'testdata.csv')
    model_path = os.path.join(prod_deployment_path, 'trainedmodel.pkl')
    # Load model
    with open(model_path, 'rb') as model_file:
        model = pickle.load(model_file)
    # Load test data
    infer_data = pd.read_csv(infer_data_path)
    features = infer_data[['lastmonth_activity',
                           'lastyear_activity',
                           'number_of_employees']]
    # Predict
    predictions = model.predict(features)
    return list(predictions)

# Function to get summary statistics


def dataframe_summary():
    """
    Calculate summary statistics for numeric columns in the dataset.

    Returns:
    - A list of dictionaries containing the mean, median, and standard
    deviation for each numeric column.
    """
    data_file_path = os.path.join(dataset_csv_path, 'finaldata.csv')
    data = pd.read_csv(data_file_path)
    numeric_columns = ['lastmonth_activity',
                       'lastyear_activity',
                       'number_of_employees']
    summary_stats = []
    for column in numeric_columns:
        stats = {
            'mean': data[column].mean(),
            'median': data[column].median(),
            'std': data[column].std()
        }
        summary_stats.append(stats)
    return summary_stats

# Missing data function


def missing_data_check():
    """
    Count missing data (NA values) in each column of the dataset and calculate
    the percentage of NA values.

    Returns:
    - A list of percentages of NA values for each column in the dataset.
    """
    dataset = pd.read_csv(os.path.join(
        config['output_folder_path'], 'finaldata.csv'))
    # Calculating the number of missing values in each column
    missing_counts = dataset.isna().sum()
    # Calculating the percentage of missing values in each column
    total_rows = len(dataset)
    missing_percentage = (missing_counts / total_rows) * 100
    # Returning the missing percentage values as a list
    return missing_percentage.tolist()

# Function to get timings


def execution_time():
    '''
    Calculate timing of training.py and ingestion.py using subprocess.
    Returns:
    - A list of 2 timing values in seconds: [ingestion_time, training_time]
    '''
    start_time = time.time()
    # Run ingestion.py and wait for it to finish
    subprocess.run(['python', 'ingestion.py'],
                   capture_output=True, text=True, check=True)
    ingestion_time = time.time() - start_time
    start_time = time.time()
    # Run training.py and wait for it to finish
    subprocess.run(['python', 'training.py'],
                   capture_output=True, text=True, check=True)
    training_time = time.time() - start_time
    return [ingestion_time, training_time]

# Function to check dependencies


def outdated_packages_list():
    """
    Print a table of outdated packages listed in requirements.txt
    with their current and latest versions.
    """
    # Read the requirements.txt file and extract package names
    with open('requirements.txt', 'r', encoding='utf-8') as f:
        required_packages = {line.split(
            '==')[0].lower() for line in f if '==' in line}

    # Execute the pip list command to get outdated packages
    result = subprocess.run(['pip', 'list', '--outdated'],
                            capture_output=True, text=True, check=True)
    lines = result.stdout.split('\n')

    # Find the start of the table headings to correctly parse package names
    for i, line in enumerate(lines):
        if 'Package' in line and 'Version' in line and 'Latest' in line:
            start_index = i + 1  # The actual data starts after the headings
            break
    else:  # If the headings are not found, there's nothing to filter/print
        print("No outdated packages found.")
        return

    # Filter and print the table headings plus outdated packages
    # listed in requirements.txt
    print(lines[start_index - 1])
    for line in lines[start_index:]:
        if line:
            package_name = line.split()[0]
            if package_name.lower() in required_packages:
                print(line)


if __name__ == '__main__':
    model_predictions()
    dataframe_summary()
    execution_time()
    outdated_packages_list()
