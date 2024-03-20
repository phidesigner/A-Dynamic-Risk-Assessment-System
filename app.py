"""
Flask application for scoring monitoring.
"""

import pickle
import json
import os
import subprocess
from flask import Flask, jsonify, request
import pandas as pd
import diagnostics


# Set up variables for use in our script
app = Flask(__name__)
app.secret_key = '1652d576-484a-49fd-913a-6879acfa6ba4'

with open('config.json', 'r', encoding='utf-8') as f:
    config = json.load(f)

dataset_csv_path = os.path.join(config['output_folder_path'])
prediction_model_path = os.path.join(
    config['production_deployment'], 'trainedmodel.pkl')

PREDICTION_MODEL = None
try:
    with open(prediction_model_path, 'rb') as model_file:
        PREDICTION_MODEL = pickle.load(model_file)
except FileNotFoundError:
    print("Model file not found. Ensure the model file path is correct in config.json.")

# Prediction Endpoint


@app.route("/prediction", methods=['POST', 'OPTIONS'])
def prediction():
    """
    Endpoint for making predictions.
    """
    if request.method == 'POST':
        data = request.get_json()
        dataset_path = data['dataset_path']
        predictions = diagnostics.model_predictions(dataset_path)
        return jsonify(predictions), 200

# Scoring Endpoint


@app.route("/scoring", methods=['GET', 'OPTIONS'])
def scoring():
    """
    Endpoint for scoring.
    """
    result = subprocess.run(['python', 'scoring.py'],
                            capture_output=True, text=True, check=True)
    score = result.stdout.strip()
    return jsonify({'score': score}), 200

# Summary Statistics Endpoint


@app.route("/summarystats", methods=['GET', 'OPTIONS'])
def summarystats():
    """
    Endpoint for summary statistics.
    """
    summary_stats = diagnostics.dataframe_summary()
    return jsonify(summary_stats), 200

# Diagnostics Endpoint


@app.route("/diagnostics", methods=['GET', 'OPTIONS'])
def diagnostics_endpoint():
    """
    Endpoint for diagnostics information.
    """
    execution_times = diagnostics.execution_time()
    missing_data = diagnostics.missing_data_check()
    outdated_packages = diagnostics.outdated_packages_list()
    diagnostics_info = {
        'execution_time': execution_times,
        'missing_data': missing_data,
        'outdated_packages': outdated_packages
    }
    return jsonify(diagnostics_info), 200


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000, debug=True, threaded=True)
