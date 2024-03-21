"""
This module contains code for making API calls and storing the responses.

The module performs the following tasks:
- Loads the configuration from a JSON file.
- Specifies the URL and test data path.
- Calls various API endpoints and stores the responses.
- Combines all API responses into a dictionary.
- Writes the combined responses to a file.

The module requires the 'config.json' file to be present in the same directory.
"""
import os
import json
import requests

# Load the configuration to get various paths
with open('config.json', 'r', encoding='utf-8') as f:
    config = json.load(f)

test_data_path = os.path.join(config['test_data_path'], 'testdata.csv')
test_data_payload = {"dataset_path": test_data_path}
output_file_path = os.path.join(config['output_model_path'], 'apireturns2.txt')

# Specify a URL that resolves to your workspace
URL = "http://127.0.0.1:8000"

# Specify the path to the test data
test_data_path = {"dataset_path": "testdata/testdata.csv"}


# Call each API endpoint and store the responses
response1 = requests.post(f"{URL}/prediction", json=test_data_payload)
response2 = requests.get(f"{URL}/scoring")
response3 = requests.get(f"{URL}/summarystats")
response4 = requests.get(f"{URL}/diagnostics")

# Combine all API responses
responses = {
    "predictions": response1.json(),
    "scoring": response2.json(),
    "summary_stats": response3.json(),
    "diagnostics": response4.json()
}

# Ensure the output directory exists
os.makedirs(config['output_model_path'], exist_ok=True)

# Write the combined responses to 'apireturns.txt'
with open(output_file_path, 'w', encoding='utf-8') as f:
    json.dump(responses, f, indent=4)
print(f"API responses written to {output_file_path}")
