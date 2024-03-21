

"""
This module contains the main process for scoring monitoring.
It checks for new data, model drift, and performs re-deployment,
diagnostics, and reporting.

The process includes the following steps:
1. Check and read new data:
    - Read the 'ingestedfiles.txt' file.
    - Determine whether the source data folder has files that aren't
      listed in 'ingestedfiles.txt'.

2. Deciding whether to proceed, part 1:
    - If new data is found, proceed with the process.
      Otherwise, end the process here.

3. Checking for model drift:
    - Check whether the score from the deployed model is different
      from the score from the model that uses the newest ingested data.

4. Deciding whether to proceed, part 2:
    - If model drift is found, proceed with the process.
      Otherwise, end the process here.

5. Re-deployment:
    - If evidence for model drift is found, re-run the 'deployment.py' script.

6. Diagnostics and reporting:
    - Run the 'diagnostics.py' and 'reporting.py' scripts for the
     re-deployed model.
"""
import os
import json
import ingestion
import training
import scoring
import deployment
import diagnostics
# import reporting

with open('config.json', 'r', encoding='utf-8') as config_file:
    config = json.load(config_file)

prod_deployment_path = config['prod_deployment_path']
input_folder_path = config['input_folder_path']
ingested_files_path = os.path.join(prod_deployment_path, 'ingestedfiles.txt')
new_data_path = os.path.join(config['output_folder_path'], 'finaldata.csv')


def main():
    """
    Main function that executes the full process of scoring monitoring.

    This function performs the following steps:
    1. Checks and reads new data.
    2. Decides whether to proceed based on the presence of new data.
    3. Checks for model drift.
    4. Decides whether to proceed with re-deployment and diagnostics based on
     model drift.
    5. Re-trains the model with new data and re-deploys the model if necessary.
    6. Runs diagnostics and reporting for the re-deployed model.
    """
    # 1. Check and read new data
    # first, read ingestedfiles.txt
    with open(ingested_files_path, 'r', encoding='utf-8') as f:
        ingested_files = f.read().splitlines()

    # Check for new data in the input folder
    source_data_files = os.listdir(input_folder_path)
    # second, determine whether the source data folder has files that aren't
    #  listed in ingestedfiles.txt
    new_files = [
        file for file in source_data_files if file not in ingested_files]

    # 2. Deciding whether to proceed, part 1
    # if you found new data, you should proceed. otherwise,
    # do end the process here
    if new_files:
        # New data found, run ingestion process
        ingestion.merge_multiple_dataframe()
        proceed = True  # Variable to control the flow
    else:
        # No new data found, end the process
        proceed = False

    # Checking for model drift
    if proceed:
        # Use the diagnostics module to generate model predictions
        predictions = diagnostics.model_predictions(new_data_path)

        # Score the new predictions
        new_score = scoring.score_model(
            predictions=predictions, new_data_path=new_data_path)

        # Read the last score from 'latestscore.txt'
        with open(os.path.join(prod_deployment_path, 'latestscore.txt'), 'r',
                  encoding='utf-8') as f:
            last_score_str = f.readline().strip()
            last_score = float(last_score_str.split(": ")[1])

        # 3. Check for model drift
        if new_score < last_score:
            model_drift = True
            print("Model drift detected")
        else:
            model_drift = False
            print("No model drift detected")

        # 4. Deciding whether to proceed, part 2
        if model_drift:
            proceed = True
            print("Proceeding with re-deployment and diagnostics")
        else:
            proceed = False
            print("Process ends due to no model drift")

        if proceed:
            # 5.1. Re-training the model with the new data
            # if you found evidence for model drift, re-run the deployment.py
            print("Re-training the model...")
            training.train_model()

            # 5.2. Re-deploying the model
            print("Re-deploying the model...")
            # Calling the function to deploy the retrained model
            deployment.store_model_into_pickle()

    # Diagnostics and reporting
    # run diagnostics.py and reporting.py for the re-deployed model


if __name__ == "__main__":
    main()
