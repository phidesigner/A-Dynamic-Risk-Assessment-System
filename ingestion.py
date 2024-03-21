"""
This module is responsible for data ingestion.
"""

import os
import json
import pandas as pd

# Load config.json and get input and output paths
with open('config.json', 'r', encoding='utf-8') as f:
    config = json.load(f)

input_folder_path = config['input_folder_path']
output_folder_path = config['output_folder_path']


# Function for data ingestion
def merge_multiple_dataframe():
    """
    Merge multiple dataframes into a single dataframe
    and write to an output file.
    """
    # check for datasets, compile them together, and write to an output file
    all_files = os.listdir(input_folder_path)
    csv_files = [file for file in all_files if file.endswith('.csv')]
    dfs = []

    for file in csv_files:
        file_path = os.path.join(input_folder_path, file)
        # Read the current file into a DataFrame
        df = pd.read_csv(file_path)
        dfs.append(df)

    # Append the data from the current file to the combined DataFrame
    combined_df = pd.concat(dfs, ignore_index=True)

    # Remove duplicate rows
    combined_df.drop_duplicates(inplace=True)

    # Write the combined and de-duplicated DataFrame to a csv file
    combined_df.to_csv(
        os.path.join(output_folder_path, 'finaldata.csv'), index=False
    )

    # Save the record of ingested files
    with open(
            os.path.join(output_folder_path, 'ingestedfiles.txt'), 'w',
            encoding='utf-8') as file_obj:
        for file in csv_files:
            file_obj.write(f"{file}\n")


if __name__ == '__main__':
    merge_multiple_dataframe()
