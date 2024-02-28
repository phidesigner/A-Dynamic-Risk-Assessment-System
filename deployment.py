# from flask import Flask, session, jsonify, request
import pandas as pd
import os
import json
import shutil


# Load config.json and correct path variable
with open('config.json', 'r') as f:
    config = json.load(f)

prod_deployment_path = os.path.join(config['prod_deployment_path'])
model_path = os.path.join(config['output_model_path'], 'trainedmodel.pkl')
score_file_path = os.path.join(config['output_model_path'], 'latestscore.txt')
output_folder_path = os.path.join(
    config['output_folder_path'], 'ingestedfiles.txt')

# function for deployment


def store_model_into_pickle():
    """
    Copies the trained model, model score, and ingested data record 
    to the production deployment directory.
    """
    try:
        # Ensure the production deployment directory exists
        if not os.path.exists(prod_deployment_path):
            os.makedirs(prod_deployment_path)

        # Copy the specified files to the production deployment directory
        shutil.copy(model_path, prod_deployment_path)
        shutil.copy(score_file_path, prod_deployment_path)
        shutil.copy(output_folder_path, prod_deployment_path)

        print("Files successfully deployed to production directory.")
    except Exception as e:
        print(f"Error during deployment: {e}")


# Example usage
if __name__ == "__main__":
    store_model_into_pickle()
