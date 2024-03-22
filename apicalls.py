import os
import json
import requests


def run_apicalls():
    """
    This function makes various API calls and stores the responses.
    """
    # Load the configuration to get various paths
    with open('config.json', 'r', encoding='utf-8') as f:
        config = json.load(f)

    output_file_path = os.path.join(
        config['output_model_path'], 'apireturns.txt')
    test_data_path = os.path.join(config['test_data_path'], 'testdata.csv')

    # Specify a URL that resolves to your workspace
    URL = "http://127.0.0.1:8000"

    # API endpoints calls
    response1 = requests.post(
        f"{URL}/prediction", json={"dataset_path": test_data_path})
    # print(response1.status_code, response1.text)
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
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

    # Write the combined responses to 'apireturns.txt'
    with open(output_file_path, 'w', encoding='utf-8') as f:
        json.dump(responses, f, indent=4)
    print(f"API responses written to {output_file_path}")


if __name__ == '__main__':
    run_apicalls()
